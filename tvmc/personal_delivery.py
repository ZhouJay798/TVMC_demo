"""Experimental session + recipient envelope using standard cryptography primitives.

Model: AES-256-GCM under independent DEK; DEK: RSA-OAEP/SHA256 under
recipient key; OAEP ciphertext: existing session HKDF/AES-GCM envelope.
Trust: authenticated registry and trusted model packager. No non-transferability.
"""
import hashlib,secrets
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric import rsa,padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from .crypto import canonical,digest,encrypt,decrypt
from .identity_gate import identity_proof,verify_identity


def sha(data):return hashlib.sha256(data).hexdigest()
def public_bytes(key):return key.public_bytes(serialization.Encoding.DER,serialization.PublicFormat.SubjectPublicKeyInfo)
def new_key():return rsa.generate_private_key(public_exponent=65537,key_size=2048)
def binding_context(admission,recipient,commitment,der):
    return {'domain':'TVMC/recipient-binding/v1','sid':admission.sid,'epoch':admission.epoch,
            'participants':list(admission.participants),'recipient':recipient,'identity_commitment':commitment,
            'recipient_key_sha256':sha(der),'admission_nonce':admission.nonce}

def bind_recipient(gate,admission,recipient,identity_secret,private_key):
    gate.require_admitted(admission)
    if recipient not in admission.participants:raise ValueError('recipient outside session')
    der=public_bytes(private_key.public_key());ctx=binding_context(admission,recipient,gate.registry[recipient],der)
    proof=identity_proof(recipient,identity_secret,ctx)
    # Publicly verifiable possession of the RSA private key. PSS and OAEP
    # use separate padding/domain contexts; no raw RSA operation is exposed.
    pop=private_key.sign(canonical(ctx),padding.PSS(mgf=padding.MGF1(hashes.SHA256()),salt_length=padding.PSS.MAX_LENGTH),hashes.SHA256())
    record={'recipient':recipient,'public_key_der':der.hex(),'identity_proof':proof,'key_possession':pop.hex()}
    verify_recipient(gate,admission,record)
    return record

def verify_recipient(gate,admission,record):
    gate.require_admitted(admission);i=record['recipient']
    if type(i) is not int or i not in admission.participants:raise ValueError('unknown recipient')
    der=bytes.fromhex(record['public_key_der']);pub=serialization.load_der_public_key(der)
    if not isinstance(pub,rsa.RSAPublicKey) or pub.key_size<2048:raise ValueError('unsupported recipient key')
    ctx=binding_context(admission,i,gate.registry[i],der)
    if record['identity_proof'].get('i')!=i or not verify_identity(gate.registry[i],record['identity_proof'],ctx):
        raise ValueError('identity/key binding failed')
    pub.verify(bytes.fromhex(record['key_possession']),canonical(ctx),padding.PSS(mgf=padding.MGF1(hashes.SHA256()),salt_length=padding.PSS.MAX_LENGTH),hashes.SHA256())
    return pub

def replica_seed(admission,record):
    i=record['recipient'];commitments=dict(admission.identity_commitments)
    return digest(['TVMC/personal-replica/v1',admission.sid,admission.epoch,f'{admission.sid}-U{i}',i,
                   commitments[i],sha(bytes.fromhex(record['public_key_der']))])

def context_for(admission,record,payload_hash):
    i=record['recipient']
    return {'domain':'TVMC/personal-delivery/v1','sid':admission.sid,'epoch':admission.epoch,
            'replica_id':f'{admission.sid}-U{i}','recipient':i,'identity_commitment':dict(admission.identity_commitments)[i],
            'recipient_key_sha256':sha(bytes.fromhex(record['public_key_der'])),
            'fingerprint_seed':replica_seed(admission,record),'payload_sha256':payload_hash}

def oaep(context):
    return padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),algorithm=hashes.SHA256(),label=hashlib.sha256(canonical(context)).digest())

def seal_copy(gate,admission,record,session_key,payload):
    pub=verify_recipient(gate,admission,record);ctx=context_for(admission,record,sha(payload))
    dek=secrets.token_bytes(32);nonce=secrets.token_bytes(12)
    ciphertext=nonce+AESGCM(dek).encrypt(nonce,payload,canonical(ctx))
    wrapctx=dict(ctx,cipher_sha256=sha(ciphertext))
    inner=pub.encrypt(dek,oaep(wrapctx));wrapped=encrypt(session_key,inner,wrapctx)
    return {'context':ctx,'ciphertext':ciphertext,'wrapped_key':wrapped}

def unwrap_inner(session_key,package,expected_context):
    if package['context']!=expected_context:raise ValueError('delivery context mismatch')
    wrapctx=dict(expected_context,cipher_sha256=sha(package['ciphertext']))
    return decrypt(session_key,package['wrapped_key'],wrapctx),wrapctx

def open_copy(session_key,private_key,package,expected_context):
    inner,wrapctx=unwrap_inner(session_key,package,expected_context)
    dek=private_key.decrypt(inner,oaep(wrapctx))
    ciphertext=package['ciphertext']
    plain=AESGCM(dek).decrypt(ciphertext[:12],ciphertext[12:],canonical(expected_context))
    if sha(plain)!=expected_context['payload_sha256']:raise ValueError('payload hash mismatch')
    return plain
