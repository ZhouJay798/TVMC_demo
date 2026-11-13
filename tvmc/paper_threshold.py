"""Research TDH2 hybrid and Pedersen/Feldman joint key-generation.

RFC3526 group14 (about 112-bit security), not the paper's ristretto255 backend.
Python arithmetic is not constant time. Local DKG aborts on invalid shares;
complaint recovery and distributed network liveness are not implemented.
"""
from dataclasses import dataclass, field
import hashlib
import secrets
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from .crypto import P, Q, G, canonical, digest, scalar, key_bytes


def point(v):
    return type(v) is int and 0 < v < P and pow(v, Q, P) == 1


def challenge(*items):
    return int(digest(['TVMC/paper-prototype/v1', *items]), 16) % Q


def hash_to_group(label):
    # Hash to a field element and square into the order-q subgroup. Never g^H(label),
    # which would expose the discrete logarithm between Pedersen generators.
    for counter in range(256):
        value=int.from_bytes(hashlib.shake_256(label+bytes([counter])).digest(256),'big')%P
        result=pow(value,2,P)
        if 1<result<P:return result
    raise ValueError('hash-to-group failed')

GBAR=hash_to_group(b'TVMC/TDH2/gbar/v2')
H_PEDERSEN=hash_to_group(b'TVMC/Pedersen/h/v2')


def verify_vss(i,value,blinding,commitments,feldman=None):
    if type(i) is not int or i<1 or not 0<=value<Q or not 0<=blinding<Q:return False
    valid=pow(G,value,P)*pow(H_PEDERSEN,blinding,P)%P==evaluate(commitments,i)
    return valid and (feldman is None or pow(G,value,P)==evaluate(feldman,i))


def evaluate(commitments, i):
    out = 1
    for k, c in enumerate(commitments):
        if not point(c):
            raise ValueError('invalid commitment')
        out = out * pow(c, pow(i, k, Q), P) % P
    return out


@dataclass(frozen=True)
class ThresholdPublic:
    n: int
    t: int
    epoch: int
    purpose: str
    commitments: tuple
    vss_transcript: tuple = ()

    @property
    def pk(self):
        return self.commitments[0]

    @property
    def kid(self):
        return digest(self.__dict__)

    def vk(self, i):
        if type(i) is not int or not 1 <= i <= self.n:
            raise ValueError('unregistered member')
        return evaluate(self.commitments, i)


@dataclass(frozen=True)
class SecretShare:
    i: int
    kid: str
    value: int = field(repr=False)


def joint_keygen(n=5,t=3,purpose='release',epoch=1):
    """Pedersen VSS commitments first, Feldman extraction second; abort on fault.

    Simulates authenticated private channels in one process. All contributions
    enter the key; does not reconstruct the aggregate secret constant.
    """
    if not 2<=t<=n or n<2*t-1:raise ValueError('require 2 <= t and n >= 2t-1')
    contributions=[];transcript=[]
    for dealer in range(1,n+1):
        a=[scalar() for _ in range(t)];b=[scalar() for _ in range(t)]
        C=[pow(G,x,P)*pow(H_PEDERSEN,y,P)%P for x,y in zip(a,b)]
        contributions.append((a,b,C))
        transcript.append(dict(dealer=dealer,pedersen=C))
    totals=[0]*n;combined=[1]*t
    for dealer,(a,b,C) in enumerate(contributions):
        A=[pow(G,x,P) for x in a]
        for i in range(1,n+1):
            value=sum(x*pow(i,k,Q) for k,x in enumerate(a))%Q
            blind=sum(x*pow(i,k,Q) for k,x in enumerate(b))%Q
            if not verify_vss(i,value,blind,C,A):raise ValueError('DKG invalid contribution; abort')
            totals[i-1]=(totals[i-1]+value)%Q
        transcript[dealer]['feldman']=A
        combined=[x*y%P for x,y in zip(combined,A)]
    public=ThresholdPublic(n,t,epoch,purpose,tuple(combined),tuple(transcript))
    if public.pk==1:raise ValueError('degenerate key')
    return public,[SecretShare(i,public.kid,x) for i,x in enumerate(totals,1)]


def weights(indices):
    if len(set(indices)) != len(indices):
        raise ValueError('duplicate members')
    return {i: _weight(i, indices) for i in indices}


def _weight(i, indices):
    v=1
    for j in indices:
        if i != j:
            v=v*(-j)*pow(i-j,-1,Q) % Q
    return v


def partial(public, share, base, context):
    if share.kid != public.kid or pow(G,share.value,P) != public.vk(share.i):
        raise ValueError('wrong key/epoch/share')
    if not point(base):
        raise ValueError('invalid base')
    d=pow(base,share.value,P)
    r=scalar(); a=pow(G,r,P); b=pow(base,r,P)
    c=challenge('CP',public.kid,share.i,public.vk(share.i),base,d,a,b,context)
    return dict(i=share.i,d=d,a=a,b=b,z=(r+c*share.value)%Q)


def verify_partial(public, base, proof, context):
    try:
        i,d,a,b,z=(proof[k] for k in ('i','d','a','b','z'))
        vk=public.vk(i)
        if not all(point(v) for v in (base,d,a,b)) or type(z) is not int or not 0<=z<Q:
            return False
        c=challenge('CP',public.kid,i,vk,base,d,a,b,context)
        return pow(G,z,P)==a*pow(vk,c,P)%P and pow(base,z,P)==b*pow(d,c,P)%P
    except (KeyError,ValueError,TypeError):
        return False


def combine(public, base, proofs, context):
    if len(proofs)<public.t:
        raise ValueError('threshold not met')
    w=weights([p['i'] for p in proofs])
    result=1
    for p in proofs:
        if not verify_partial(public,base,p,context):
            raise ValueError('invalid partial decryption')
        result=result*pow(p['d'],w[p['i']],P)%P
    return result


def td_mask(shared):
    return hashlib.sha256(b'TVMC/TDH2/H1/v2'+shared.to_bytes(256,'big')).digest()


def td_challenge(public,cap,w,wbar):
    return challenge('TDH2/H2',public.kid,cap['label'],cap['a'],cap['abar'],
                     cap['c'],cap['nonce'],cap['ct'],w,wbar)


def seal(public,payload,label):
    """TDH2 encrypts a fresh 32-byte AEAD key; label binds its wrapped payload.

    Equations follow Shoup-Gennaro TDH2; distinct encoding/group from upstream.
    This is a tested research port, not an audited implementation/security proof.
    """
    key=secrets.token_bytes(32);nonce=secrets.token_bytes(12)
    aad=canonical([public.kid,label])
    ct=AESGCM(key).encrypt(nonce,payload,aad)
    r=scalar();k=scalar();a=pow(G,r,P);abar=pow(GBAR,r,P)
    c=bytes(x^y for x,y in zip(key,td_mask(pow(public.pk,r,P))))
    cap=dict(kid=public.kid,label=label,a=a,abar=abar,c=c.hex(),nonce=nonce.hex(),ct=ct.hex())
    e=td_challenge(public,cap,pow(G,k,P),pow(GBAR,k,P))
    return dict(cap,e=e,z=(k+e*r)%Q)


def check_capsule(public,cap,label):
    try:
        if cap['kid']!=public.kid or cap['label']!=label:return False
        if not all(point(cap[k]) and cap[k]!=1 for k in ('a','abar')):return False
        e,z=cap['e'],cap['z']
        if not all(type(v) is int and 0<=v<Q for v in (e,z)):return False
        if len(bytes.fromhex(cap['c']))!=32 or len(bytes.fromhex(cap['nonce']))!=12:return False
        if len(bytes.fromhex(cap['ct']))<16:return False
        w=pow(G,z,P)*pow(pow(cap['a'],e,P),-1,P)%P
        wbar=pow(GBAR,z,P)*pow(pow(cap['abar'],e,P),-1,P)%P
        return e==td_challenge(public,cap,w,wbar)
    except (KeyError,ValueError,TypeError):return False


def open_capsule(public,cap,proofs,label,context):
    if not check_capsule(public,cap,label):raise ValueError('invalid TDH2 capsule')
    shared=combine(public,cap['a'],proofs,context)
    key=bytes(x^y for x,y in zip(bytes.fromhex(cap['c']),td_mask(shared)))
    return AESGCM(key).decrypt(bytes.fromhex(cap['nonce']),bytes.fromhex(cap['ct']),canonical([public.kid,label]))


def refresh(public, shares):
    """Simulate proactive zero-sharing refresh; public encryption key unchanged.

    Caller must securely replace and erase old shares. No mobile-adversary claim
    for Python memory; identity-key rotation and membership changes are separate.
    """
    if sorted(s.i for s in shares)!=list(range(1,public.n+1)):
        raise ValueError('refresh requires all current members in this simulation')
    values={s.i:s.value for s in shares}
    for s in shares:
        if s.kid!=public.kid or pow(G,s.value,P)!=public.vk(s.i):raise ValueError('invalid old share')
    commitments=list(public.commitments)
    for _ in range(public.n):
        polynomial=[0]+[scalar() for _ in range(public.t-1)]
        updates=[pow(G,a,P) for a in polynomial]
        for i in values:
            piece=sum(a*pow(i,k,Q) for k,a in enumerate(polynomial))%Q
            if pow(G,piece,P)!=evaluate(updates,i):raise ValueError('invalid refresh piece')
            values[i]=(values[i]+piece)%Q
        commitments=[a*b%P for a,b in zip(commitments,updates)]
    new=ThresholdPublic(public.n,public.t,public.epoch+1,public.purpose,tuple(commitments))
    return new,[SecretShare(i,new.kid,values[i]) for i in sorted(values)]
