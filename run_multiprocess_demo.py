"""Run issuer, registrar and five custodians as separate local processes.

The coordinator is a trusted local dealer and recovery service. Identity/RSA
private keys stay in custodian workers, and Ed25519 signing keys stay in issuer
and registrar workers, but the coordinator sees the shares and session secret.
All workers run under the same OS user and use local files/JSON pipes: this is
not production isolation, DKG, threshold decryption, or an independent witness.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
import secrets
import tempfile
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

from cryptography.exceptions import InvalidSignature, InvalidTag

from tvmc.bundle_delivery import open_bundle
from tvmc.crypto import G, P, Q, scalar
from tvmc.epochs import EpochPublic
from tvmc.identity_gate import IdentityGate
from tvmc.personal_delivery import replica_seed
from tvmc.process_recovery import reconstruct_submissions
from tvmc.process_report import write_process_report
from tvmc.process_roles import RoleError, RoleProcess
from tvmc.reconstruction import request_context, verify_receipt
from tvmc.tardos import Tardos
from tvmc.trusted_registry import code_commitment, verify_issuance, verify_snapshot


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    path.chmod(0o600)


def rejects(action):
    try:
        action()
    except (ValueError, RoleError, InvalidTag, InvalidSignature):
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new private output directory')
    parser.add_argument('--offline-fixture', action='store_true', help='synthetic model and text; integration only')
    parser.add_argument('--model-path', type=Path, help='local causal model directory for a real model run')
    parser.add_argument('--text-corpus', type=Path, help='local plaintext corpus; otherwise use cached WikiText-2')
    parser.add_argument('--cache', type=Path, default=Path('.cache/huggingface'))
    parser.add_argument('--steps', type=int, default=1200, help='fingerprint training steps per model')
    args = parser.parse_args()
    if args.offline_fixture == bool(args.model_path):
        parser.error('choose exactly one of --offline-fixture or --model-path')
    if args.offline_fixture and args.text_corpus:
        parser.error('--text-corpus cannot be combined with --offline-fixture')
    if args.steps < 1:
        parser.error('--steps must be positive')
    from tvmc.process_payloads import detect_payload
    from tvmc.offline_fixture import make_fixture

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    fixture_data = None
    if args.offline_fixture:
        source = out / 'fixture_source'
        fixture_data = make_fixture(source)
    else:
        source = args.model_path.resolve()
    weights = source / 'model.safetensors'
    if not weights.is_file():
        raise FileNotFoundError(f'missing model weights: {weights}')
    base_hash = sha_file(weights)
    ids = [1, 2, 3]
    session_id = 'MP-E0'

    with ExitStack() as stack:
        issuer = stack.enter_context(RoleProcess('issuer'))
        admin = stack.enter_context(RoleProcess('admin'))
        custodians = {i: stack.enter_context(RoleProcess('custodian', custodian_id=i)) for i in range(1, 6)}
        processes = {'coordinator': os.getpid(), 'issuer': issuer.info['pid'], 'registrar': admin.info['pid'],
                     **{f'U{i}': custodians[i].info['pid'] for i in custodians}}
        if len(set(processes.values())) != len(processes):
            raise RuntimeError('roles are not running in distinct processes')
        print('PROCESSES', json.dumps(processes), flush=True)

        enrollment = [custodians[i].call('enrollment') for i in custodians]
        gate = IdentityGate(enrollment)
        nonce = secrets.token_hex(32)
        admission_context = gate.context(session_id, 0, ids, base_hash, nonce)
        proofs = [custodians[i].call('identity_proof', context=admission_context) for i in ids]
        admission = gate.register(session_id, 0, ids, base_hash, nonce, proofs)
        admission_data = asdict(admission)
        recipients = [custodians[i].call('bind_recipient', admission=admission_data) for i in ids]
        print('ADMISSION', 'five preregistered; U1/U2/U3 admitted with identity and RSA key proofs', flush=True)

        tardos = Tardos(n=3)
        words = [tardos.encode(replica_seed(admission, record)) for record in recipients]
        replica_ids = [f'{session_id}-U{i}' for i in ids]
        registered = admin.call('create_precommit', enrollment=enrollment, admission=admission_data,
                                recipients=recipients)
        precommit, openings = registered['precommit'], registered['openings']
        pre_body = verify_snapshot(precommit, admin.public_key())
        if any(item['code_commitment'] != code_commitment(rid, word, openings[rid])
               for item, rid, word in zip(pre_body['entries'], replica_ids, words)):
            raise ValueError('registrar precommit does not match the prepared fingerprints')
        dump(out / 'registry_precommit.json', precommit)

        # The local coordinator is a trusted dealer: it creates all shares and
        # knows the session secret. This is not distributed key generation.
        coefficients = [scalar() for _ in ids]
        session_secret = coefficients[0]
        public = EpochPublic(tuple(pow(G, value, P) for value in coefficients),
                             tuple(gate.registry[i] for i in range(1, 6)), 5, 3, tuple(ids))
        for i in ids:
            value = sum(a * pow(i, degree, Q) for degree, a in enumerate(coefficients)) % Q
            custodians[i].call('store_share', value=value, public=asdict(public))
        del coefficients, value
        key_context = {'domain': 'TVMC/admitted-key-recovery/v1', 'sid': admission.sid, 'epoch': admission.epoch,
            'participants': list(admission.participants), 'threshold': 3, 'seed': admission.seed,
            'base_sha256': admission.base_sha256, 'nonce': admission.nonce}
        dump(out / 'public_registration.json', {
            'enrollment': enrollment, 'admission': admission_data, 'recipients': recipients,
            'key_public': asdict(public), 'key_context': key_context, 'replica_ids': replica_ids,
            'issuer_public': issuer.info['public_key'], 'registrar_public': admin.info['public_key'],
            'warning': 'local simulation keys are not independently authenticated trust anchors'})

        print('TRAINING', 'three independently fingerprinted models; this can take several minutes', flush=True)
        plaintext_temp = tempfile.TemporaryDirectory(prefix='.issuer_plain_', dir=out)
        stack.callback(plaintext_temp.cleanup)
        built = issuer.call('build_payloads', output_dir=plaintext_temp.name,
                            words=[word.tolist() for word in words], replica_ids=replica_ids,
                            offline_fixture=args.offline_fixture, model_path=str(source),
                            fixture_data=fixture_data,
                            text_corpus=str(args.text_corpus.resolve()) if args.text_corpus else None,
                            cache=str(args.cache.resolve()), steps=args.steps)
        if built['base_sha256'] != base_hash:
            raise ValueError('prepared models do not use the admitted base weights')
        dump(out / 'audit_codebook.json', {'replica_ids': replica_ids, 'words': [w.tolist() for w in words],
             'p': tardos.p.tolist(), 'threshold': tardos.threshold, 'openings': openings,
             'warning': 'restricted demo data; fixed demo seed and local plaintext are not production security'})
        dump(out / 'audit_prompts.json', built['prompts'])

        print('ISSUING', 'issuer process encrypts the three packages and signs the manifest', flush=True)
        issued_package = issuer.call('make_and_seal_files', enrollment=enrollment, admission=admission_data,
            recipients=recipients, session_secret=session_secret,
            payload_paths=[r['path'] for r in built['replicas']], outer_path=str(out / 'outer_bundle.aesgcm'))
        plaintext_temp.cleanup()
        manifest = issued_package['manifest']
        signature = issued_package['signature']
        if sha_file(out / 'outer_bundle.aesgcm') != issued_package['outer_sha256']:
            raise ValueError('saved outer ciphertext hash does not match issuer result')
        issued = admin.call('create_issuance', enrollment=enrollment, admission=admission_data,
            recipients=recipients, precommit=precommit, manifest=manifest, issuer_signature=signature,
            issuer_public=issuer.info['public_key'])
        verify_issuance(precommit, issued, admin.public_key(), manifest, bytes.fromhex(signature),
                        issuer.public_key(), gate, admission, recipients)
        dump(out / 'registry_issued.json', issued)
        dump(out / 'signed_manifest.json', {'manifest': manifest, 'signature': signature,
             'issuer_public': issuer.info['public_key'], 'registrar_public': admin.info['public_key']})

        context = request_context(public, key_context, issued_package['outer_sha256'], manifest, secrets.token_hex(32))
        submissions = [custodians[i].call('prove_share', context=context) for i in ids]
        checks = {
            'distinct_processes': True,
            'insufficient_shares_rejected': rejects(lambda: reconstruct_submissions(public, submissions[:2], context, issuer)),
            'duplicate_shares_rejected': rejects(lambda: reconstruct_submissions(public, [submissions[0]] * 3, context, issuer)),
            'cross_request_proofs_rejected': rejects(lambda: reconstruct_submissions(public, submissions,
                                                        dict(context, request_nonce=secrets.token_hex(32)), issuer)),
        }
        wrong_share = copy.deepcopy(submissions)
        wrong_share[0]['value'] = (wrong_share[0]['value'] + 1) % Q
        checks['modified_share_rejected'] = rejects(lambda: reconstruct_submissions(public, wrong_share, context, issuer))
        restored, receipt = reconstruct_submissions(public, submissions, context, issuer)
        checks['all_three_recover_session_secret'] = restored == session_secret
        verify_receipt(receipt, public, context, issuer.public_key())
        dump(out / 'reconstruction_receipt.json', receipt)
        # This coordinator sees outer plaintext and all submitted share values;
        # the role private keys and identity secrets never cross these pipes.
        print('RECOVERING', 'the trusted coordinator receives three share values and opens the outer package', flush=True)
        outer = (out / 'outer_bundle.aesgcm').read_bytes()
        entries, _ = open_bundle(gate, admission, recipients, outer, restored, issuer.public_key())
        del outer, submissions, wrong_share, restored, session_secret
        transit = out / 'encrypted_transit'
        transit.mkdir(mode=0o700)
        file_entries = []
        for i, entry in enumerate(entries, 1):
            cipher = transit / f'U{i}.cipher'; wrapped = transit / f'U{i}.key'
            cipher.write_bytes(entry['ciphertext']); cipher.chmod(0o600)
            wrapped.write_bytes(entry['wrapped_key']); wrapped.chmod(0o600)
            file_entries.append({'context': entry['context'], 'cipher_path': str(cipher), 'wrapped_path': str(wrapped)})
        del entries
        gc.collect()
        common = {'enrollment': enrollment, 'admission': admission_data, 'recipients': recipients,
                  'precommit': precommit, 'issued': issued, 'manifest': manifest,
                  'issuer_signature': signature, 'issuer_public': issuer.info['public_key'],
                  'registry_public': admin.info['public_key']}
        wrong_issuer = dict(common, issuer_signature='00' * 64)
        altered_registry = copy.deepcopy(issued)
        altered_registry['body']['entries'][0]['recipient'] = 2
        wrong_registry = dict(common, issued=altered_registry)
        checks['invalid_issuer_signature_rejected'] = rejects(lambda: custodians[1].call('verified_open', entry=file_entries[0], **wrong_issuer))
        checks['tampered_registry_rejected'] = rejects(lambda: custodians[1].call('verified_open', entry=file_entries[0], **wrong_registry))
        wrong_owner = []
        for owner in ids:
            for other in ids:
                if owner != other:
                    wrong_owner.append(rejects(lambda owner=owner, other=other: custodians[other].call(
                        'verified_open', entry=file_entries[owner - 1], **common)))
        checks['all_six_wrong_recipient_packages_rejected'] = all(wrong_owner)

        rows = []
        for i in ids:
            target = out / 'custodians' / f'U{i}' / 'received_model.zip'
            print('RECIPIENT', f'U{i} verifies both signed records and decrypts their model', flush=True)
            opened = custodians[i].call('verified_open', entry=file_entries[i - 1], output_path=str(target), **common)
            same = opened['payload_sha256'] == built['replicas'][i - 1]['sha256'] == sha_file(target)
            print('TRACING', f'U{i} decrypted model is evaluated with the registered probes', flush=True)
            trace = detect_payload(zip_path=target, prompts=built['prompts'], words=words,
                                   replica_ids=replica_ids, cache=args.cache)
            rows.append({'recipient': i, 'replica_id': replica_ids[i - 1], 'payload_sha256_matches': same,
                         'payload_path': opened['payload_path'], 'trace': trace})
        checks['all_owner_model_hashes_match'] = all(row['payload_sha256_matches'] for row in rows)
        checks['all_models_trace_to_their_copy'] = all(row['trace']['decoded_replicas'] == [row['replica_id']] for row in rows)
        metrics = {'status': 'completed' if all(checks.values()) else 'completed_with_failed_checks',
                   'processes': processes, 'participants': ids, 'base_sha256': base_hash,
                   'model_scope': built['scope'], 'training_steps_per_model': args.steps,
                   'model_source': str(source),
                   'text_corpus_sha256': sha_file(args.text_corpus) if args.text_corpus else None,
                   'checks': checks, 'wrong_recipient_rejections': wrong_owner, 'recipients': rows,
                   'limitations': [
                       'The coordinator is a trusted dealer and recovery service; it sees all share values and the session secret.',
                       'The issuer worker trains and encrypts packages; the coordinator performs local fingerprint detection.',
                       'Workers run under one OS user; inherited pipes and owner-only paths do not provide adversarial isolation.',
                       'Identity/RSA private keys and issuer/registrar signing keys are generated in workers and not exported via the role API.',
                       'Roles exit after the run; private keys and shares are not persisted, so the ciphertext cannot be reopened later.',
                       'The registrar process rebuilds and verifies the admission before signing both precommit and issuance.',
                       'No independent witness, trusted clock, authenticated external identity, durable replay prevention, DKG or threshold decryption.',
                       'Fingerprint training and detection use the same probes; results do not establish unseen-probe robustness.',
                       'Local plaintext ZIPs and audit codebook are sensitive; same-UID workers can access them.',
                       'The signed receipt is produced using the issuer key in this demo, not an independently operated recovery service.'
                   ]}
        dump(out / 'metrics.json', metrics)
        write_process_report(out / 'MULTIPROCESS_REPORT.md', metrics)
        print('COMPLETE', json.dumps({'status': metrics['status'], 'checks': checks}, ensure_ascii=False), flush=True)
        return 0 if all(checks.values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
