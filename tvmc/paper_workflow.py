"""New-paper custody path, isolated from the legacy API.

Local protocol simulation, CA-validated Ed25519 identities and X25519 recipients,
hash-chain ledger (not independent transparency witnesses). See migration docs.
"""
import copy
import hashlib
import json
import secrets
from dataclasses import dataclass, field
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from .crypto import canonical, digest
from . import paper_hpke as hpke
from . import paper_threshold as te
from .pki import DemoCA, Trust, Registry, pem, member_key, issuer_key


def _ciphertext_digest(package):
    from .stream_delivery import ciphertext_digest
    return ciphertext_digest(package)


def sha(data):
    from pathlib import Path
    if isinstance(data,Path):
        from .stream_delivery import file_hash
        return file_hash(data)
    return hashlib.sha256(data).hexdigest()


def sign(sk,kind,obj):
    return sk.sign(canonical(['TVMC/paper/v1',kind,obj])).hex()


def verify(pk,kind,obj,sig):
    try:
        Ed25519PublicKey.from_public_bytes(pk).verify(bytes.fromhex(sig),canonical(['TVMC/paper/v1',kind,obj]))
        return True
    except Exception:
        # cryptography.InvalidSignature is deliberately treated as rejection.
        return False


@dataclass
class Member:
    i: int
    identity: object = field(default_factory=Ed25519PrivateKey.generate,repr=False)
    receiving: object = field(default_factory=X25519PrivateKey.generate,repr=False)

    certificates: dict = field(default_factory=dict,repr=False)

    @property
    def registration(self):
        return dict(i=self.i,identity=self.identity.public_key().public_bytes_raw().hex(),
                    receiving=self.receiving.public_key().public_bytes_raw().hex(),**self.certificates)


class Ledger:
    def __init__(self):
        self.entries=[]

    def append(self,kind,body):
        e=dict(seq=len(self.entries),prev=self.entries[-1]['hash'] if self.entries else '0'*64,
               kind=kind,body=copy.deepcopy(body))
        e['hash']=digest(e); self.entries.append(e)
        return e['seq']

    def check(self,expected_head=None):
        prev='0'*64
        for i,e in enumerate(self.entries):
            if e['seq']!=i or e['prev']!=prev or e['hash']!=digest({k:v for k,v in e.items() if k!='hash'}):
                return False
            prev=e['hash']
        return expected_head is None or prev==expected_head


class Custody:
    def __init__(self,n=5,t=3,ledger=None):
        self.members=[Member(i) for i in range(1,n+1)]
        self.ledger=ledger if ledger is not None else Ledger(); self.registry={}; self.manifests={}; self.requests={}; self.used=set()
        self.issuer=Ed25519PrivateKey.generate()
        self._issuer_pk=self.issuer.public_key().public_bytes_raw()
        self.ca=DemoCA()
        issuer_cert=self.ca.issue('distributor',self.issuer.public_key(),'issuer')
        self.registry=Registry(Trust(pem(self.ca.root),self.ca.crl(),issuer_cert))
        for m in self.members:
            m.certificates=dict(identity_certificate=self.ca.issue('U'+str(m.i),m.identity.public_key(),'identity'),
                                receiving_certificate=self.ca.issue('U'+str(m.i),m.receiving.public_key(),'receiving'))
            reg=m.registration; sig=sign(m.identity,'join',reg)
            self.registry[m.i]=copy.deepcopy(reg)
            challenge=secrets.token_bytes(32)
            wrapped=hpke.seal(member_key(self.registry,reg['i'],'receiving'),challenge,canonical(reg))
            if hpke.open(m.receiving,wrapped,canonical(reg))!=challenge or not verify(member_key(self.registry,reg['i']),'join',reg,sig):
                raise ValueError('registration failed')
            self.registry[m.i]=copy.deepcopy(reg)
            self.ledger.append('registration',dict(registration=reg,signature=sig))
        self.release_key,self.release_shares=te.joint_keygen(n,t,'release')
        self.code_key,self.code_shares=te.joint_keygen(n,t,'codebook')
        self.ledger.append('key_generation',dict(release=self.release_key.__dict__,codebook=self.code_key.__dict__))

    @property
    def issuer_pk(self):
        return issuer_key(self.registry,self._issuer_pk)

    def issue(self,recipient,payload,cid=None):
        self.issuer_pk
        member_key(self.registry,recipient)

        reg=self.registry[recipient]; cid=cid or secrets.token_hex(16)
        if cid in self.manifests:
            raise ValueError('copy identifier already issued')
        key=secrets.token_bytes(32); omega=secrets.token_bytes(32)
        chunks=[]; size=4*1024*1024; total=max(1,(len(payload)+size-1)//size)
        for j in range(total):
            aad=canonical([cid,j,total,j==total-1])
            chunks.append(AESGCM(key).encrypt(j.to_bytes(12,'big'),payload[j*size:(j+1)*size],aad).hex())
        ct_hash=digest(chunks)
        inner=hpke.seal(member_key(self.registry,reg['i'],'receiving'),key+omega,cid.encode())
        label=[cid,ct_hash]; cap=te.seal(self.release_key,inner,label)
        code_entries=[e['body'] for e in self.ledger.entries if e['kind']=='encrypted_codeword' and e['body']['cid']==cid]
        msg=dict(cid=cid,recipient=recipient,registration=reg,commitment=sha(payload+omega),
                 codebook_hash=digest(code_entries),
                 ciphertext_hash=ct_hash,capsule_hash=digest(cap),release_kid=self.release_key.kid)
        manifest=dict(message=msg,signature=sign(self.issuer,'manifest',msg))
        self.manifests[cid]=copy.deepcopy(manifest)
        self.ledger.append('manifest',manifest)
        return dict(manifest=manifest,chunks=chunks,capsule=cap)

    def validate_package(self,package):
        man=package['manifest']; msg=man['message']; cid=msg['cid']
        member_key(self.registry,msg['recipient'])
        member_key(self.registry,msg['recipient'],'receiving')
        if (self.manifests.get(cid)!=man or not verify(self.issuer_pk,'manifest',msg,man['signature']) or
            msg['registration']!=self.registry.get(msg['recipient']) or
            _ciphertext_digest(package)!=msg['ciphertext_hash'] or digest(package['capsule'])!=msg['capsule_hash'] or
            not te.check_capsule(self.release_key,package['capsule'],[cid,msg['ciphertext_hash']])):
            raise ValueError('invalid manifest/package/recipient binding')
        return msg

    def request(self,member,package):
        msg=self.validate_package(package)
        req=dict(cid=msg['cid'],recipient=member.i,capsule_hash=msg['capsule_hash'],
                 epoch=self.release_key.epoch,kid=self.release_key.kid,nonce=secrets.token_hex(16),policy='any-t-registered')
        signed=dict(request=req,signature=sign(member.identity,'request',req))
        self.validate_request(signed,package)
        self.requests[digest(req)]=copy.deepcopy(signed)
        self.ledger.append('request',signed)
        return signed

    def validate_request(self,signed,package):
        msg=self.validate_package(package); req=signed['request']; i=req['recipient']
        if (i!=msg['recipient'] or req['cid']!=msg['cid'] or req['capsule_hash']!=msg['capsule_hash'] or
            req['epoch']!=self.release_key.epoch or req['kid']!=self.release_key.kid or
            req['policy']!='any-t-registered' or not req['nonce'] or
            not verify(member_key(self.registry,i),'request',req,signed['signature'])):
            raise ValueError('invalid release request')
        return digest(req)

    def intent(self,member,signed,package):
        rid=self.validate_request(signed,package)
        if rid not in self.requests or rid in self.used:
            raise ValueError('unknown or completed request')
        intent=dict(i=member.i,rid=rid)
        result=dict(intent,signature=sign(member.identity,'intent',intent))
        if not verify(member_key(self.registry,member.i),'intent',intent,result['signature']):
            raise ValueError('unregistered identity key')
        self.ledger.append('intent',result)
        return result

    def valid_intents(self,rid):
        approved=set()
        for e in self.ledger.entries:
            if e['kind']!='intent': continue
            body=e['body']; i=body['i']
            if body['rid']==rid and i in self.registry and verify(member_key(self.registry,i),
                      'intent',dict(i=i,rid=rid),body['signature']): approved.add(i)
        return approved

    def approval(self,member,signed,package):
        rid=self.validate_request(signed,package)
        intents=self.valid_intents(rid)
        if rid in self.used or len(intents)<self.release_key.t or member.i not in intents:
            raise ValueError('phase one incomplete or replay')
        context=['release',rid,digest(package['capsule'])]
        proof=te.partial(self.release_key,self.release_shares[member.i-1],package['capsule']['a'],context)
        body=dict(rid=rid,proof=proof)
        return dict(body,signature=sign(member.identity,'approval',body))

    def release(self,signed,package,approvals):
        rid=self.validate_request(signed,package)
        intents=self.valid_intents(rid)
        if rid not in self.requests or rid in self.used or len(intents)<self.release_key.t:
            raise ValueError('unknown request, replay or missing intent quorum')
        for approval in approvals:
            i=approval['proof']['i']; body={k:v for k,v in approval.items() if k!='signature'}
            if i not in intents or approval['rid']!=rid or not verify(member_key(self.registry,i),
                        'approval',body,approval['signature']): raise ValueError('invalid approval signature')
        inner=te.open_capsule(self.release_key,package['capsule'],[a['proof'] for a in approvals],
                  [signed['request']['cid'],_ciphertext_digest(package)],['release',rid,digest(package['capsule'])])
        self.used.add(rid); self.ledger.append('release',dict(rid=rid,approvals=approvals))
        return inner

    def receive(self,member,package,inner):
        msg=self.validate_package(package); cid=msg['cid']
        if msg['registration']!=member.registration:
            raise ValueError('wrong recipient')
        secret=hpke.open(member.receiving,inner,cid.encode())
        if len(secret)!=64: raise ValueError('invalid wrapped payload')
        key,omega=secret[:32],secret[32:]; chunks=package['chunks']; plaintext=[]
        for j,ct in enumerate(chunks):
            plaintext.append(AESGCM(key).decrypt(j.to_bytes(12,'big'),bytes.fromhex(ct),canonical([cid,j,len(chunks),j==len(chunks)-1])))
        payload=b''.join(plaintext)
        if sha(payload+omega)!=msg['commitment']: raise ValueError('copy commitment mismatch')
        receipt=dict(cid=cid,ciphertext_hash=msg['ciphertext_hash'],received=True)
        self.ledger.append('receipt',dict(receipt=receipt,signature=sign(member.identity,'receipt',receipt)))
        return payload
