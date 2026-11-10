"""Verify a signed local registry against caller-supplied trusted public keys.

The caller must authenticate the trust-anchor file and the checkpoint challenge
through an independent channel. --historical replays at the timestamp asserted
by the witness; it does not establish trusted time or present-day freshness.
This audit does not prove that a model was actually trained with its codeword,
that a remote server ran the model, or that a particular person leaked it.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature
from tvmc.identity_gate import IdentityGate
from tvmc.epochs import EpochPublic
from tvmc.personal_delivery import replica_seed
from tvmc.reconstruction import request_context, verify_receipt
from tvmc.robustness import aggregate
from tvmc.tardos import Tardos
from tvmc.trusted_registry import (verify_issuance, verify_snapshot, verify_checkpoint,
                                   verify_recipient_binding, code_commitment)


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def audit_directory(directory, trusted_anchors, expected_challenge, *, historical=False, now=None):
    root=Path(directory)
    anchors=_load(trusted_anchors)
    issuer=Ed25519PublicKey.from_public_bytes(bytes.fromhex(anchors['issuer']))
    registry=Ed25519PublicKey.from_public_bytes(bytes.fromhex(anchors['registry']))
    witness=Ed25519PublicKey.from_public_bytes(bytes.fromhex(anchors['witness']))
    reg=_load(root/'registration_before_training.json')
    old=_load(root/'registry_precommit.json')
    old_cp=_load(root/'registry_precommit_checkpoint.json')
    issued=_load(root/'registry_issued.json')
    checkpoint=_load(root/'registry_checkpoint.json')
    signed=_load(root/'signed_manifest.json')
    if signed['issuer_public_key'] != anchors['issuer']:
        raise ValueError('issuer key in signed record does not match trusted anchor')
    admission_data=reg['admission']
    gate=IdentityGate(reg['enrollment'])
    admission=gate.register(admission_data['sid'],admission_data['epoch'],admission_data['participants'],
                            admission_data['base_sha256'],admission_data['nonce'],reg['request_proofs'])
    if admission.seed != admission_data['seed']:
        raise ValueError('admission seed does not match the registration')
    issuer_signature=bytes.fromhex(signed['signature_hex'])
    body=verify_issuance(old,issued,registry,signed['manifest'],issuer_signature,issuer,
                         gate,admission,reg['recipients'])
    pre_body=verify_snapshot(old,registry)
    # Historical precommit signature is checked at the time asserted by this
    # witness. The witness is local in the demo; this is not trusted timestamping.
    verify_checkpoint(old_cp,witness,old,registry,old_cp['body']['challenge'],now=old_cp['body']['issued_at'])
    check_time=checkpoint['body']['issued_at'] if historical else (int(time.time()) if now is None else now)
    verify_checkpoint(checkpoint,witness,issued,registry,expected_challenge,now=check_time)
    openings=_load(root/'fingerprint_openings.json')
    if (len(reg['replica_ids']) != len(pre_body['entries']) or
        len(reg['words']) != len(pre_body['entries']) or
        len(reg['recipients']) != len(pre_body['entries']) or
        len(set(reg['replica_ids'])) != len(pre_body['entries'])):
        raise ValueError('registration count or replica uniqueness mismatch')
    tardos=Tardos(n=len(reg['recipients']))
    scheme=pre_body['fingerprint_scheme']
    if (scheme['p'] != tardos.p.tolist() or reg['p'] != scheme['p'] or
        scheme['threshold'] != tardos.threshold or reg['threshold'] != scheme['threshold']):
        raise ValueError('fingerprint decoding parameters differ from signed precommit')
    for rid,word,recipient in zip(reg['replica_ids'],reg['words'],reg['recipients']):
        record=verify_recipient_binding(body,rid,recipient['recipient'],bytes.fromhex(recipient['public_key_der']))
        if tardos.encode(replica_seed(admission,recipient)).tolist()!=word:
            raise ValueError('fingerprint code does not match the registered seed')
        if code_commitment(rid,word,openings[rid])!=record['code_commitment']:
            raise ValueError('fingerprint code differs from signed precommit')
    receipt=_load(root/'reconstruction_receipt.json')
    public=EpochPublic(**reg['key_public'])
    metrics=_load(root/'model_metrics.json')
    cipher_path=root/'outer_bundle.aesgcm'
    actual_hash=_file_hash(cipher_path) if cipher_path.is_file() else metrics['outer_cipher_sha256']
    context=request_context(public,reg['key_context'],actual_hash,signed['manifest'],
                            receipt['record']['context']['request_nonce'])
    verify_receipt(receipt,public,context,issuer)
    trace_paths=[root/(replica+'_batch_http.json') for replica in reg['replica_ids']]
    trace_results=[]
    if any(path.is_file() for path in trace_paths):
        if not all(path.is_file() for path in trace_paths):
            raise ValueError('incomplete trace transcripts')
        words=np.asarray(reg['words'])
        for index,path in enumerate(trace_paths):
            trace=_load(path)
            values=trace['response']['score_differences']
            if len(values)!=768 or not np.isfinite(values).all():
                raise ValueError('invalid trace scores')
            observed=aggregate(values,3)
            scores=tardos.scores(words,observed)
            decoded=[reg['replica_ids'][i] for i in tardos.decode(words,observed)]
            if decoded!=trace['result']['decoded_replicas']:
                raise ValueError('HTTP transcript and trace result disagree')
            trace_results.append({'tested_replica':reg['replica_ids'][index],
                                  'matched_bits':int((observed==words[index]).sum()),
                                  'scores':scores.tolist(),'threshold':tardos.threshold,
                                  'decoded_replicas':decoded})
    return {'issuer_signature_and_registry_binding':True,'code_commitments_valid':True,
            'checkpoint_signature_and_chain_valid':True,
            'checkpoint_freshness':'not_established_historical_replay' if historical else 'valid_at_local_clock',
            'reconstruction_receipt_valid':True,
            'outer_ciphertext_hash_rechecked':cipher_path.is_file(),
            'trace_transcript_consistency':trace_results,
            'warning':'Trust anchors and expected challenge must be authenticated independently. '
                      'Local demo keys and witness are not independent, and HTTP/model provenance is not established.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--trusted-anchors',type=Path,required=True,help='independently authenticated issuer/registry/witness public keys')
    p.add_argument('--challenge',required=True,help='expected witness challenge obtained through a trusted channel')
    p.add_argument('--historical',action='store_true',help='replay at asserted witness time; does not prove freshness')
    args=p.parse_args()
    try:
        result=audit_directory(args.run_dir,args.trusted_anchors,args.challenge,historical=args.historical)
    except (ValueError,KeyError,TypeError,FileNotFoundError,InvalidSignature) as exc:
        p.exit(2,f'audit failed: {type(exc).__name__}: {exc}\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
