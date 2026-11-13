"""Bounded-memory model delivery. Manifest binds ordered ciphertext chunk hashes."""
import hashlib
import os
from pathlib import Path
import secrets
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from .crypto import canonical,digest
from .paper_workflow import sign
from . import paper_hpke as hpke,paper_threshold as te

CHUNK=4*1024*1024


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(CHUNK),b''):h.update(block)
    return h.hexdigest()


def ciphertext_digest(package):
    if package.get('storage')=='files':return digest([file_hash(p) for p in package['chunks']])
    return digest(package['chunks'])


def issue_file(custody,recipient,path,directory,cid):
    if cid in custody.manifests:raise ValueError('duplicate cid')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False,mode=0o700)
    reg=custody.registry[recipient];key=secrets.token_bytes(32);omega=secrets.token_bytes(32)
    count=max(1,(Path(path).stat().st_size+CHUNK-1)//CHUNK);chunks=[];h=hashlib.sha256()
    with open(path,'rb') as src:
        for j in range(count):
            data=src.read(CHUNK);h.update(data)
            encrypted=AESGCM(key).encrypt(j.to_bytes(12,'big'),data,canonical([cid,j,count,j==count-1]))
            target=directory/f'{j:08d}.bin';target.write_bytes(encrypted);chunks.append(str(target))
    h.update(omega)
    package=dict(storage='files',chunks=chunks)
    ct_hash=ciphertext_digest(package)
    inner=hpke.seal(bytes.fromhex(reg['receiving']),key+omega,cid.encode())
    cap=te.seal(custody.release_key,inner,[cid,ct_hash])
    codes=[e['body'] for e in custody.ledger.entries if e['kind']=='encrypted_codeword' and e['body']['cid']==cid]
    msg=dict(cid=cid,recipient=recipient,registration=reg,commitment=h.hexdigest(),
             ciphertext_hash=ct_hash,capsule_hash=digest(cap),codebook_hash=digest(codes),release_kid=custody.release_key.kid)
    manifest=dict(message=msg,signature=sign(custody.issuer,'manifest',msg))
    import copy
    custody.manifests[cid]=copy.deepcopy(manifest);custody.ledger.append('manifest',manifest)
    package.update(manifest=manifest,capsule=cap)
    return package


def receive_file(custody,member,package,inner,target):
    msg=custody.validate_package(package)
    if msg['registration']!=member.registration:raise ValueError('wrong recipient')
    secret=hpke.open(member.receiving,inner,msg['cid'].encode())
    if len(secret)!=64:raise ValueError('wrong inner length')
    key,omega=secret[:32],secret[32:];target=Path(target)
    if target.exists():raise ValueError('do not overwrite received artifact')
    target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    tmp=target.with_name(target.name+'.'+secrets.token_hex(8)+'.partial');h=hashlib.sha256();cid=msg['cid']
    try:
        fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as dest:
            count=len(package['chunks'])
            for j,p in enumerate(package['chunks']):
                data=AESGCM(key).decrypt(j.to_bytes(12,'big'),Path(p).read_bytes(),canonical([cid,j,count,j==count-1]))
                dest.write(data);h.update(data)
            dest.flush();os.fsync(dest.fileno())
        h.update(omega)
        if h.hexdigest()!=msg['commitment']:raise ValueError('plaintext commitment mismatch')
        os.replace(tmp,target)
    finally:
        if tmp.exists():tmp.unlink()
    receipt=dict(cid=cid,ciphertext_hash=msg['ciphertext_hash'],received=True)
    custody.ledger.append('receipt',dict(receipt=receipt,signature=sign(member.identity,'receipt',receipt)))
    return target
