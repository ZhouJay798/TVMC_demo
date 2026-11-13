"""Additive research backend for the new paper's threshold workflow.

Feldman joint key-generation simulation over the existing RFC3526 group.
This is NOT the paper's malicious-secure Pedersen DKG or TDH2 implementation.
No constant-time guarantee; all actors share a Python address space in this demo.
Secret shares are never exported by the public transcript or combine operation.
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


def joint_keygen(n=5, t=3, purpose='release', epoch=1):
    """All dealers contribute independent polynomials; verify every delivered evaluation.

    Local simulation: no authenticated network, complaints or disqualification.
    Does not compute or return the aggregate constant term (complete private key).
    """
    if not 2 <= t <= n or n < 2*t-1:
        raise ValueError('require 2 <= t <= n and n >= 2t-1')
    totals = [0]*n
    combined = [1]*t
    for _ in range(n):
        polynomial = [scalar() for _ in range(t)]
        commitments = [pow(G, a, P) for a in polynomial]
        for i in range(1, n+1):
            piece = sum(a*pow(i, k, Q) for k, a in enumerate(polynomial)) % Q
            if pow(G, piece, P) != evaluate(commitments, i):
                raise ValueError('invalid dealer contribution')
            totals[i-1] = (totals[i-1]+piece) % Q
        combined = [a*b % P for a,b in zip(combined, commitments)]
    public = ThresholdPublic(n,t,epoch,purpose,tuple(combined))
    if public.pk == 1:
        raise ValueError('degenerate public key; rerun key generation')
    return public, [SecretShare(i,public.kid,s) for i,s in enumerate(totals,1)]


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


def seal(public, payload, label):
    """Labeled DH-KEM+AEAD prototype with ephemeral-key knowledge proof; NOT TDH2.

    No claim of paper-equivalent CCA security. Verify before providing partials.
    """
    r=scalar(); a=pow(G,r,P)
    nonce=secrets.token_bytes(12)
    aad=canonical([public.kid,label,a])
    ct=AESGCM(key_bytes(pow(public.pk,r,P))).encrypt(nonce,payload,aad)
    cap=dict(kid=public.kid,label=label,a=a,nonce=nonce.hex(),ct=ct.hex())
    k=scalar(); commitment=pow(G,k,P)
    c=challenge('capsule',cap,commitment)
    return dict(cap,proof={'a':commitment,'z':(k+c*r)%Q})


def check_capsule(public, cap, label):
    try:
        if cap['kid']!=public.kid or cap['label']!=label or not point(cap['a']) or cap['a']==1:
            return False
        proof=cap['proof']; bare={k:v for k,v in cap.items() if k!='proof'}
        z=proof['z']; a=proof['a']
        c=challenge('capsule',bare,a)
        return (point(a) and type(z) is int and 0<=z<Q and
                pow(G,z,P)==a*pow(cap['a'],c,P)%P)
    except (KeyError,ValueError,TypeError):
        return False


def open_capsule(public, cap, proofs, label, context):
    if not check_capsule(public,cap,label):
        raise ValueError('invalid capsule')
    shared=combine(public,cap['a'],proofs,context)
    return AESGCM(key_bytes(shared)).decrypt(bytes.fromhex(cap['nonce']),bytes.fromhex(cap['ct']),
                                            canonical([public.kid,label,cap['a']]))


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
