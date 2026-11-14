"""Member-side Algorithm 6 guard. No arbitrary ciphertext decryption API.

Trusted provisioning supplies one share, identity key, public roots and evidence
reader. The coordinator cannot supply a different reader in a request. This local
prototype still co-locates keys; it is not an isolated service or durable policy.
"""
import copy
import math
from .crypto import P, G, digest
from .paper_workflow import Ledger, sha, sign, verify
from . import paper_threshold as te


def score_ciphertext(params, observed, ciphertexts):
    if len(observed) != len(params['biases']) or len(ciphertexts) != len(observed):
        raise ValueError('score length mismatch')
    scale = params['scale']
    if type(scale) is not int or scale <= 0:
        raise ValueError('invalid scale')
    a, b, offset = 1, 1, 0.
    for p, y, ct in zip(params['biases'], observed, ciphertexts):
        if not 0 < p < 1 or type(y) is not int or y not in (-1, 0, 1):
            raise ValueError('invalid score input')
        u, v = math.sqrt((1-p)/p), math.sqrt(p/(1-p))
        coefficient = round(scale * (0 if y == -1 else (u+v)*(1 if y == 1 else -1)))
        offset += 0 if y == -1 else (-v if y == 1 else v)
        a = a * pow(ct['a'], coefficient, P) % P
        b = b * pow(ct['b'], coefficient, P) % P
    return dict(a=a, b=b * pow(G, round(scale*offset), P) % P)


class ScoreApprover:
    def __init__(self, public, share, identity, issuer_pk, registry, evidence_reader):
        self.public, self.share, self.identity = public, share, identity
        self.issuer_pk, self.registry = issuer_pk, copy.deepcopy(registry)
        self.reader = evidence_reader
        self.used_groups = set()

    def approve(self, request, entries, trusted_head, artifact):
        """Validate the entire batch before producing ANY partial decryption.

        trusted_head must come from the member's trusted checkpoint channel, not
        an untrusted request. Evidence reader must be locally configured. Exact
        retries are also rejected; policy state must be persisted in deployment.
        """
        from .paper_trace import verify_bit
        req, entries = copy.deepcopy(request), copy.deepcopy(entries)
        group = req['group']
        if group in self.used_groups:
            raise ValueError('score group already used')
        if req['kid'] != self.public.kid:
            raise ValueError('wrong scoring key or checkpoint')
        ledger = Ledger(); ledger.entries = entries
        if not ledger.check(trusted_head):
            raise ValueError('untrusted ledger snapshot')

        def one(kind, predicate):
            matches = [e for e in entries if e['kind'] == kind and predicate(e['body'])]
            if len(matches) != 1:
                raise ValueError('missing or duplicate ' + kind)
            return matches[0]

        evidence = one('evidence_commitment', lambda b: b['group'] == group)
        disclosure = one('group_disclosed_and_retired', lambda b: b['parameters']['group'] == group)
        commitment = one('parameters_commitment', lambda b: b['group'] == group)
        logged_request = one('score_request', lambda b: b['group'] == group)
        if (logged_request['body'] != req or logged_request['prev'] != req['head'] or
            not disclosure['seq'] < logged_request['seq'] or
            any(e['kind'] == 'trace_result' and e['body']['group'] == group for e in entries)):
            raise ValueError('unregistered or previous scoring request')
        if not commitment['seq'] < evidence['seq'] < disclosure['seq']:
            raise ValueError('invalid evidence/disclosure ordering')
        if req['evidence'] != sha(artifact) or evidence['body']['file_sha256'] != req['evidence']:
            raise ValueError('evidence substitution')
        params = disclosure['body']['parameters']
        if commitment['body']['commitment'] != digest([params, disclosure['body']['salt']]):
            raise ValueError('parameter substitution')
        if params['reader']['kind'] == 'signed-api-logit-difference':
            from .crypto import canonical
            from .evidence import verify_quorum
            endpoint = one('endpoint_commitment', lambda b: b['group'] == group)
            probes_open = one('probe_disclosure', lambda b: b['group'] == group)
            escrow = one('probe_escrow', lambda b: b['group'] == group)
            collected = one('endpoint_evidence', lambda b: b['group'] == group)
            target = endpoint['body']['target']; opening = probes_open['body']['opening']
            probes = opening['parameters']['probes']
            if (sha(canonical(target)) != req['evidence'] or
                not escrow['seq'] < evidence['seq'] < probes_open['seq'] < collected['seq'] < disclosure['seq'] or
                not endpoint['seq'] < probes_open['seq'] or
                probes_open['body']['evidence_hash'] != req['evidence'] or
                escrow['body']['commitment'] != digest([opening['parameters'], opening['salt']]) or
                params['reader']['probe_hash'] != digest(probes)):
                raise ValueError('API probe/target binding mismatch')
            if any(e['kind']=='legitimate_endpoint' and e['body']['endpoint']==target['endpoint'] for e in entries):
                raise ValueError('registered legitimate endpoint')
            if params['reader']['delta'] != 1.:
                raise ValueError('unsupported API readout margin')
            observed = verify_quorum(collected['body']['records'], self.registry, self.public.t,
                                     target, probes, group, params['reader']['repeats'])
        else:
            observed = list(self.reader(artifact))
        if observed != req['observed']:
            raise ValueError('observation not supported by evidence')
        codes = [e for e in entries if e['kind'] == 'encrypted_codeword' and e['body']['group'] == group]
        if not codes or len({e['body']['cid'] for e in codes}) != len(codes):
            raise ValueError('invalid codebook')
        expected = []
        for entry in codes:
            cid, cts = entry['body']['cid'], entry['body']['ciphertexts']
            manifest = one('manifest', lambda b: b['message']['cid'] == cid)
            man = manifest['body']; msg = man['message']
            if not commitment['seq'] < entry['seq'] < manifest['seq'] < evidence['seq']:
                raise ValueError('codebook not registered before evidence')
            all_codes = [e['body'] for e in entries if e['kind'] == 'encrypted_codeword' and e['body']['cid'] == cid]
            if (not verify(self.issuer_pk, 'manifest', msg, man['signature']) or
                msg['codebook_hash'] != digest(all_codes) or
                msg['registration'] != self.registry.get(msg['recipient'])):
                raise ValueError('invalid signed issuance binding')
            if not all(verify_bit(self.public, ct, [group, cid, k]) for k, ct in enumerate(cts)):
                raise ValueError('invalid codeword proof')
            expected.append(dict(cid=cid, **score_ciphertext(params, observed, cts)))
        if req['scores'] != expected:
            raise ValueError('submitted ciphertext is not the recomputed score batch')
        # Reserve before generating responses; fail closed on signing errors.
        self.used_groups.add(group)
        responses = []
        for row in expected:
            ctx = ['score', group, req['evidence'], observed, row['cid'], row['a'], row['b']]
            proof = te.partial(self.public, self.share, row['a'], ctx)
            signature = sign(self.identity, 'score', dict(context=ctx, proof=proof))
            responses.append(dict(cid=row['cid'], proof=proof, signature=signature))
        approval = dict(request_hash=digest(req), member=self.share.i)
        return dict(approval=approval, approval_signature=sign(self.identity, 'score-approval', approval), responses=responses)
