"""Verify the exported transcript using only an independently trusted public context."""
import argparse
import json
from pathlib import Path
import numpy as np
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from tvmc.crypto import Public, verify_proof
from tvmc.tardos import Tardos
from tvmc.bridge import Registry, Record, judge


def verify(context, evidence):
    public = Public(**context['public'])
    registry = Registry()
    for row in context['records']:
        rec = Record(**row)
        registry.records[rec.sid] = rec
    tardos = Tardos(length=len(context['tardos_p']))
    tardos.p = np.array(context['tardos_p'])
    tardos.threshold = context['threshold']
    key_ok = (len(context['key_proofs']) >= public.threshold and
              len({p['i'] for p in context['key_proofs']}) == len(context['key_proofs']) and
              all(verify_proof(public, p, context['key_context']) for p in context['key_proofs']))
    dep_ok = judge(registry, public, tardos, evidence,
                   Ed25519PublicKey.from_public_bytes(bytes.fromhex(context['collector_key'])),
                   context['endpoint'], context['nonce'], context['now'])
    rec = registry.records.get(evidence.get('sid'))
    proof_ids = {p['i'] for p in context['key_proofs']}
    bridge_ok = bool(key_ok and dep_ok and rec and
                     rec.sid == context['key_context']['sid'] and
                     rec.epoch == context['key_context']['epoch'] and
                     proof_ids == set(rec.participants))
    return {'key_layer': bool(key_ok), 'deployment_layer': bool(dep_ok), 'bridge_layer': bridge_ok}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    context = json.loads((args.run_dir/'public_context.json').read_text(encoding='utf-8'))
    evidence = json.loads((args.run_dir/'public_evidence.json').read_text(encoding='utf-8'))
    result = verify(context, evidence)
    print(json.dumps(result, indent=2))
    return 0 if all(result.values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
