"""Whole-bundle envelope for individually encrypted 3/3-session replicas.

Requires a trusted pinned issuer Ed25519 public key and trusted registry. This
local prototype does not establish an issuer's real identity or secure devices.
"""
import io,json,secrets,zipfile
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from .crypto import canonical,encrypt,decrypt
from .personal_delivery import sha,verify_recipient,context_for,oaep


def make_inner(gate,admission,record,payload):
    pub=verify_recipient(gate,admission,record);ctx=context_for(admission,record,sha(payload))
    dek=secrets.token_bytes(32);nonce=secrets.token_bytes(12)
    ciphertext=nonce+AESGCM(dek).encrypt(nonce,payload,canonical(ctx))
    wrapping=dict(ctx,cipher_sha256=sha(ciphertext))
    wrapped=pub.encrypt(dek,oaep(wrapping))
    return {'context':ctx,'ciphertext':ciphertext,'wrapped_key':wrapped}


def outer_context(admission):
    return {'domain':'TVMC/whole-bundle/v1','sid':admission.sid,'epoch':admission.epoch,
            'participants':list(admission.participants),'admission_seed':admission.seed,
            'base_sha256':admission.base_sha256,'admission_nonce':admission.nonce}


def seal_bundle(gate,admission,records,entries,session_key,issuer):
    gate.require_admitted(admission)
    if len(records)!=len(admission.participants) or len(entries)!=len(records):raise ValueError('incorrect bundle size')
    manifest={'version':1,'session':outer_context(admission),'entries':[]}
    for index,(record,entry) in enumerate(zip(records,entries)):
        verify_recipient(gate,admission,record);ctx=entry['context']
        if ctx!=context_for(admission,record,ctx['payload_sha256']):raise ValueError('wrong context')
        manifest['entries'].append({'context':ctx,'cipher_sha256':sha(entry['ciphertext']),'wrapped_sha256':sha(entry['wrapped_key']),
                                    'cipher_path':f'{index}.cipher','wrapped_path':f'{index}.key'})
    if {x['context']['recipient'] for x in manifest['entries']}!=set(admission.participants):raise ValueError('wrong participants')
    manifest_bytes=canonical(manifest);signature=issuer.sign(manifest_bytes)
    buff=io.BytesIO()
    with zipfile.ZipFile(buff,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        z.writestr('manifest.json',manifest_bytes);z.writestr('manifest.sig',signature)
        for spec,entry in zip(manifest['entries'],entries):
            z.writestr(spec['cipher_path'],entry['ciphertext']);z.writestr(spec['wrapped_path'],entry['wrapped_key'])
    plaintext=buff.getvalue();buff.close()
    wrapped=encrypt(session_key,plaintext,outer_context(admission))
    return wrapped,manifest,signature


def open_bundle(gate,admission,records,wrapped,session_key,issuer_public):
    gate.require_admitted(admission);raw=decrypt(session_key,wrapped,outer_context(admission));entries=[]
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        manifest_bytes=z.read('manifest.json');issuer_public.verify(z.read('manifest.sig'),manifest_bytes)
        manifest=json.loads(manifest_bytes)
        if canonical(manifest)!=manifest_bytes or manifest['session']!=outer_context(admission) or manifest['version']!=1:raise ValueError('manifest mismatch')
        by_id={r['recipient']:r for r in records}
        if set(by_id)!=set(admission.participants) or len(records)!=len(by_id) or len(manifest['entries'])!=len(by_id):raise ValueError('recipient mismatch')
        seen=set()
        for index,item in enumerate(manifest['entries']):
            ctx=item['context'];i=ctx['recipient']
            if i in seen or i not in by_id or item['cipher_path']!=f'{index}.cipher' or item['wrapped_path']!=f'{index}.key':raise ValueError('entry mismatch')
            verify_recipient(gate,admission,by_id[i])
            if ctx!=context_for(admission,by_id[i],ctx['payload_sha256']):raise ValueError('recipient binding mismatch')
            ciphertext=z.read(item['cipher_path']);inner=z.read(item['wrapped_path'])
            if sha(ciphertext)!=item['cipher_sha256'] or sha(inner)!=item['wrapped_sha256']:raise ValueError('entry hash mismatch')
            entries.append({'context':ctx,'ciphertext':ciphertext,'wrapped_key':inner});seen.add(i)
    return entries,manifest


def open_inner(entry,private_key):
    ctx=entry['context'];ciphertext=entry['ciphertext'];wrapping=dict(ctx,cipher_sha256=sha(ciphertext))
    dek=private_key.decrypt(entry['wrapped_key'],oaep(wrapping))
    plain=AESGCM(dek).decrypt(ciphertext[:12],ciphertext[12:],canonical(ctx))
    if sha(plain)!=ctx['payload_sha256']:raise ValueError('plaintext hash mismatch')
    return plain
