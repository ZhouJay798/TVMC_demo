"""Public verification against a trusted append-only registry snapshot.

A signed collector receipt prevents editing a measured transcript. It does not
prove an untrusted collector or endpoint is honest. This explicit demo trust
assumption is necessary: a copied fingerprint alone is not provenance proof.
"""
import secrets
from dataclasses import dataclass
import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from .crypto import canonical, digest


def identity_seed(sid, epoch, participants, public):
    return digest(['TVMC/identity/v1', sid, epoch,
                   [[i, public.identities[i - 1]] for i in sorted(participants)]])


def commitment(word, opening):
    return digest(['TVMC/word/v1', list(map(int, word)), opening])


@dataclass
class Record:
    sid: str
    epoch: int
    participants: list
    seed: str
    cm_w: str
    expires: int
    consumer: int


class Registry:
    def __init__(self):
        self.records = {}

    def register(self, sid, epoch, participants, public, tardos, expires, consumer):
        if sid in self.records or len(set(participants)) != len(participants):
            raise ValueError('duplicate session/participant')
        if not participants or any(i < 1 or i > public.n for i in participants):
            raise ValueError('invalid participants')
        seed = identity_seed(sid, epoch, participants, public)
        word = tardos.encode(seed)
        opening = secrets.token_hex(32)
        rec = Record(sid, epoch, sorted(participants), seed,
                     commitment(word, opening), expires, consumer)
        self.records[sid] = rec
        return rec, word, opening


class Collector:
    def __init__(self):
        self._key = Ed25519PrivateKey.generate()
        self.public_key = self._key.public_key()

    def collect(self, box, prompts, protocol, endpoint, nonce):
        observed = box.query(prompts, protocol)
        receipt = {'endpoint': endpoint, 'nonce': nonce, 'protocol': protocol,
                   'observed': list(map(int, observed))}
        return receipt, self._key.sign(canonical(receipt)).hex()


class OneShotBox:
    """One request total; internal batching is explicit, never a hidden reset."""
    def __init__(self, predictor):
        self.predictor, self.used = predictor, False

    def query(self, prompts, protocol):
        if self.used:
            raise ValueError('one-shot box already consumed')
        self.used = True
        if protocol == 'batch':
            return self.predictor(prompts)
        if protocol == 'stream':
            # One request, many frames; invokes the model on each chunk.
            parts = [self.predictor(prompts[k:k+32]) for k in range(0, len(prompts), 32)]
            return np.concatenate(parts)
        if protocol == 'single':
            return self.predictor(prompts[:1])
        raise ValueError('unknown protocol')


def judge(registry, public, tardos, evidence, collector_key, endpoint, nonce, now):
    try:
        receipt = evidence['receipt']
        collector_key.verify(bytes.fromhex(evidence['signature']), canonical(receipt))
        if receipt['endpoint'] != endpoint or receipt['nonce'] != nonce:
            return False
        observed = receipt['observed']
        if len(observed) != tardos.length or not set(observed) <= {0, 1}:
            return False  # insufficient evidence is not an accusation
        rec = registry.records[evidence['sid']]
        if now >= rec.expires or evidence['seed'] != rec.seed:
            return False
        if rec.seed != identity_seed(rec.sid, rec.epoch, rec.participants, public):
            return False
        word = evidence['word']
        if len(word) != tardos.length or not set(word) <= {0, 1}:
            return False
        if commitment(word, evidence['opening']) != rec.cm_w:
            return False
        if evidence['cm_w'] != rec.cm_w:
            return False
        return bool(tardos.scores([word], observed)[0] >= tardos.threshold)
    except (InvalidSignature, KeyError, ValueError, TypeError):
        return False


def evidence_for(rec, word, opening, receipt, signature):
    return {'sid': rec.sid, 'seed': rec.seed, 'cm_w': rec.cm_w,
            'word': list(map(int, word)), 'opening': opening,
            'receipt': receipt, 'signature': signature}
