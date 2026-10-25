"""Shamir (2,5), Feldman VSS, AND-Schnorr Fiat-Shamir proof, AES-GCM.

RFC 3526 group 14, order q=(p-1)/2, generator 2. Python big integers
are not constant time. The proof establishes knowledge of a share AND its
identity secret, bound to session/epoch/asset; it is not a general SNARK.
"""
import hashlib
import json
import secrets
from dataclasses import dataclass, replace

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


@dataclass(frozen=True)
class Share:
    i: int
    value: int
    gamma: int


@dataclass(frozen=True)
class Public:
    commitments: tuple
    identities: tuple
    n: int = 5
    threshold: int = 2

    def y(self, i):
        if not 1 <= i <= self.n:
            raise ValueError('unknown custodian index')
        return self.commitments[0] * pow(self.commitments[1], i, P) % P

    def verify(self, share):
        return (1 <= share.i <= self.n and 0 <= share.value < Q and
                0 < share.gamma < Q and pow(G, share.value, P) == self.y(share.i)
                and pow(G, share.gamma, P) == self.identities[share.i - 1])


def share_key():
    k, a = scalar(), scalar()
    shares = [Share(i, (k + a * i) % Q, scalar()) for i in range(1, 6)]
    public = Public((pow(G, k, P), pow(G, a, P)),
                    tuple(pow(G, s.gamma, P) for s in shares))
    return k, shares, public


def challenge(public, i, a, b, context):
    return int(digest(['TVMC/AND-Schnorr/v1', public.commitments,
                       public.identities, i, public.y(i), a, b, context]), 16) % Q


def prove(public, share, context):
    if not public.verify(share):
        raise ValueError('invalid share')
    r, s = scalar(), scalar()
    a, b = pow(G, r, P), pow(G, s, P)
    c = challenge(public, share.i, a, b, context)
    return {'i': share.i, 'a': a, 'b': b,
            'zf': (r + c * share.value) % Q, 'zg': (s + c * share.gamma) % Q}


def verify_proof(public, proof, context):
    try:
        i, a, b, zf, zg = [proof[k] for k in ('i', 'a', 'b', 'zf', 'zg')]
        if not all(type(v) is int for v in (i, a, b, zf, zg)):
            return False
        if not (1 <= i <= public.n and 0 <= zf < Q and 0 <= zg < Q):
            return False
        if not all(1 < v < P and pow(v, Q, P) == 1 for v in (a, b)):
            return False
        c = challenge(public, i, a, b, context)
        return (pow(G, zf, P) == a * pow(public.y(i), c, P) % P and
                pow(G, zg, P) == b * pow(public.identities[i - 1], c, P) % P)
    except (KeyError, ValueError, TypeError):
        return False


def recover(public, shares, proofs, context):
    if len(shares) < public.threshold:
        raise ValueError('need at least two shares')
    if len({s.i for s in shares}) != len(shares):
        raise ValueError('duplicate shares')
    if len(proofs) != len(shares):
        raise ValueError('missing NIZK')
    for s, proof in zip(shares, proofs):
        if not public.verify(s) or proof.get('i') != s.i or not verify_proof(public, proof, context):
            raise ValueError('share/NIZK verification failed')
    k = 0
    for s in shares:
        weight = 1
        for other in shares:
            if s.i != other.i:
                weight = weight * (-other.i) * pow(s.i - other.i, -1, Q) % Q
        k = (k + weight * s.value) % Q
    return k


def encrypt(k, payload, aad):
    nonce = secrets.token_bytes(12)
    return nonce + AESGCM(key_bytes(k)).encrypt(nonce, payload, canonical(aad))


def decrypt(k, ciphertext, aad):
    return AESGCM(key_bytes(k)).decrypt(ciphertext[:12], ciphertext[12:], canonical(aad))
