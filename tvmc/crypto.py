"""Shared finite-group parameters and canonical encodings for TVMC v2.
RFC3526 group14; Python arithmetic is research-only, not constant time.
"""
import hashlib
import json
import secrets

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

P = int('''FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1
29024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD
3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44
C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4
B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A6916
3FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966
D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C1
80E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183
995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF'''.replace('\n', ''), 16)
Q = (P - 1) // 2
G = 2


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def scalar():
    return secrets.randbelow(Q - 1) + 1


def key_bytes(k):
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=b'TVMC/AES-256-GCM/v1').derive(k.to_bytes(256, 'big'))
