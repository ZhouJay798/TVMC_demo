"""RFC 9180 base mode: X25519/HKDF-SHA256/AES-256-GCM, single message.
No PSK/auth mode. Kept separate from all legacy RSA delivery modules.
"""
import hashlib
import hmac
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEM=b'KEM'+bytes.fromhex('0020')
SUITE=b'HPKE'+bytes.fromhex('002000010002')


def extract(salt,ikm):
    return hmac.new(salt or bytes(32),ikm,hashlib.sha256).digest()


def expand(prk,info,length):
    result=b''; last=b''
    for i in range(1,(length+31)//32+1):
        last=hmac.new(prk,last+info+bytes([i]),hashlib.sha256).digest(); result+=last
    return result[:length]


def le(suite,salt,label,ikm):
    return extract(salt,b'HPKE-v1'+suite+label+ikm)


def lx(suite,prk,label,info,length):
    return expand(prk,length.to_bytes(2,'big')+b'HPKE-v1'+suite+label+info,length)


def schedule(dh,enc,pk,info):
    ss=lx(KEM,le(KEM,b'',b'eae_prk',dh),b'shared_secret',enc+pk,32)
    context=b'\x00'+le(SUITE,b'',b'psk_id_hash',b'')+le(SUITE,b'',b'info_hash',info)
    secret=le(SUITE,ss,b'secret',b'')
    return lx(SUITE,secret,b'key',context,32),lx(SUITE,secret,b'base_nonce',context,12)


def seal(pk,plaintext,info):
    ephemeral=X25519PrivateKey.generate(); enc=ephemeral.public_key().public_bytes_raw()
    key,nonce=schedule(ephemeral.exchange(X25519PublicKey.from_public_bytes(pk)),enc,pk,info)
    return enc+AESGCM(key).encrypt(nonce,plaintext,b'')


def open(sk,wrapped,info):
    enc=wrapped[:32]; pk=sk.public_key().public_bytes_raw()
    key,nonce=schedule(sk.exchange(X25519PublicKey.from_public_bytes(enc)),enc,pk,info)
    return AESGCM(key).decrypt(nonce,wrapped[32:],b'')
