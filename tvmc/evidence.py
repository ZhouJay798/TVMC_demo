"""API evidence quorum and anchored served-query rebuttal.

Collectors sign raw responses; all decoded words must be identical. Local role
simulation does not establish that an arbitrary remote service is immutable.
"""
from .pki import member_key

import hashlib
import time
from .crypto import digest,canonical
from .paper_workflow import sign,verify


def read_scores(scores,repeats=3,delta=1.):
    import math
    if type(repeats) is not int or repeats<1 or not math.isfinite(delta) or delta<=0:raise ValueError('invalid readout parameters')
    if len(scores)%repeats or any(not math.isfinite(float(x)) for x in scores):raise ValueError('unreadable response')
    word=[]
    for k in range(0,len(scores),repeats):
        total=sum(scores[k:k+repeats])
        if not math.isfinite(total):raise ValueError('nonfinite aggregate readout')
        word.append(1 if total>=delta else 0 if total<=-delta else -1)
    return word


def collect_quorum(custody,target,probes,query,indices,group,repeats=3):
    """query(target, probes, member_id) must access the pinned target itself.

    Target contains endpoint, TLS certificate SHA256, start/end wall-clock window.
    The function is injectable for tests; the HTTP adapter enforces certificate pin.
    """
    if len(indices)<custody.release_key.t or len(set(indices))!=len(indices):raise ValueError('collector quorum missing')
    if not all(k in target for k in ('endpoint','tls_sha256','start','end')):raise ValueError('incomplete target')
    if not any(e['kind']=='endpoint_commitment' and e['body']==dict(group=group,target=target) for e in custody.ledger.entries):
        raise ValueError('endpoint commitment missing')
    if any(e['kind'] in ('endpoint_evidence','endpoint_collection_started') and e['body']['group']==group for e in custody.ledger.entries):raise ValueError('group already collected or consumed')
    for i in indices:
        if i not in custody.registry:raise ValueError('unknown collector')
        member_key(custody.registry,i)
    # Consume before sending any probe. Failed or late collections cannot silently
    # retry a disclosed group. Journal persistence is local single-writer only.
    custody.ledger.append('endpoint_collection_started',dict(group=group,target=target,collectors=list(indices)))
    records=[]
    for i in indices:
        if i not in custody.registry:raise ValueError('unknown collector')
        now=time.time()
        if not target['start']<=now<=target['end']:raise ValueError('outside committed time window')
        scores=query(target,probes,i)
        finished=time.time()
        if not now<=finished<=target['end']:raise ValueError('response completed outside committed time window')
        body=dict(member=i,group=group,target=target,probe_hash=digest(probes),scores=scores,
                  time=now,finished=finished,evidence_version=2)
        records.append(dict(body=body,signature=sign(custody.members[i-1].identity,'api-evidence',body)))
    word=verify_quorum(records,custody.registry,custody.release_key.t,target,probes,group,repeats)
    custody.ledger.append('endpoint_evidence',dict(group=group,records=records,word=word))
    return word,records


def verify_quorum(records,registry,threshold,target,probes,group,repeats=3):
    ids=set();words=[];versions=set()
    for entry in records:
        b=entry['body'];i=b['member']
        if i in ids or i not in registry:raise ValueError('duplicate or unknown collector')
        if b['target']!=target or b['group']!=group or b['probe_hash']!=digest(probes):raise ValueError('evidence binding mismatch')
        version=b.get('evidence_version',1)
        if type(version) is not int or version not in (1,2):raise ValueError('unsupported evidence version')
        versions.add(version)
        if not target['start']<=b['time']<=target['end']:raise ValueError('outside window')
        if version==2 and not b['time']<=b.get('finished',float('nan'))<=target['end']:
            raise ValueError('response completed outside committed time window')
        if len(b['scores'])!=len(probes):raise ValueError('missing responses')
        if not verify(member_key(registry,i),'api-evidence',b,entry['signature']):raise ValueError('invalid collector signature')
        ids.add(i);words.append(read_scores(b['scores'],repeats))
    if len(versions)!=1:raise ValueError('mixed evidence versions')
    if len(ids)<threshold:raise ValueError('insufficient collectors')
    if any(w!=words[0] for w in words):raise ValueError('collectors disagree; insufficient evidence')
    return words[0]


def sketch(text,width=64):
    words=text.lower().split();grams=[' '.join(words[j:j+3]) for j in range(max(1,len(words)-2))]
    if not grams:grams=['']
    return [min(int.from_bytes(hashlib.sha256(canonical(['TVMC/minhash',seed,g])).digest()[:8],'big') for g in grams) for seed in range(width)]


class ServedLog:
    def __init__(self,member,ledger):self.member=member;self.ledger=ledger;self.entries=[]

    def record(self,text,client,when=None):
        e=dict(seq=len(self.entries),prev=self.entries[-1]['hash'] if self.entries else '0'*64,
               content_hash=digest(text),client=client,time=time.time() if when is None else when,sketch=sketch(text))
        e['hash']=digest(e);self.entries.append(e)

    def anchor(self):
        head=dict(member=self.member.i,size=len(self.entries),head=self.entries[-1]['hash'] if self.entries else '0'*64,time=time.time())
        anchor=dict(head=head,signature=sign(self.member.identity,'served-head',head))
        self.ledger.append('served-head',anchor);return anchor

    def rebut(self,probes,window,threshold=.8):
        matches=[]
        for k,p in enumerate(probes):
            s=sketch(p['text'])
            for e in self.entries:
                if window[0]<=e['time']<=window[1] and (e['content_hash']==digest(p['text']) or sum(a==b for a,b in zip(s,e['sketch']))/len(s)>=threshold):
                    matches.append(dict(probe=k,entry=e['seq']));break
        # Whole chain is provided in this research implementation; private sparse
        # Merkle openings are a future privacy improvement, explicitly not claimed.
        return dict(member=self.member.i,entries=self.entries,matches=matches)


def verify_rebuttal(rebuttal,anchor,registry,probes,window,delta=60.,similarity=.8):
    member=rebuttal['member'];head=anchor['head']
    if member!=head['member'] or not verify(member_key(registry,member),'served-head',head,anchor['signature']):raise ValueError('invalid anchored head')
    entries=rebuttal['entries'];prev='0'*64
    if head['size']!=len(entries):raise ValueError('chain length mismatch')
    for i,e in enumerate(entries):
        if e['seq']!=i or e['prev']!=prev or e['hash']!=digest({k:v for k,v in e.items() if k!='hash'}):raise ValueError('chain tampered')
        prev=e['hash']
    if prev!=head['head']:raise ValueError('head mismatch')
    positions=set()
    for match in rebuttal['matches']:
        k=match['probe'];e=entries[match['entry']];p=probes[k];s=sketch(p['text'])
        if not window[0]<=e['time']<=window[1] or not 0<=head['time']-e['time']<=delta:raise ValueError('late anchor')
        if e['content_hash']!=digest(p['text']) and sum(a==b for a,b in zip(s,e['sketch']))/len(s)<similarity:raise ValueError('unmatched probe')
        positions.add(p['bit'])
    return sorted(positions)


def https_query(target,probes,member):
    """Actual TLS-pinned HTTP transport, private collector to committed endpoint."""
    import http.client,ssl,json
    from urllib.parse import urlsplit
    url=urlsplit(target['endpoint'])
    if url.scheme!='https':raise ValueError('HTTPS required')
    # Certificate authentication is by explicit SHA256 pin, not public CA.
    ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT);ctx.check_hostname=False;ctx.verify_mode=ssl.CERT_NONE
    conn=http.client.HTTPSConnection(url.hostname,url.port or 443,context=ctx,timeout=60)
    try:
        conn.connect()
        if hashlib.sha256(conn.sock.getpeercert(binary_form=True)).hexdigest()!=target['tls_sha256']:raise ValueError('TLS pin mismatch')
        body=json.dumps(dict(probes=probes,collector=member)).encode()
        conn.request('POST',url.path or '/',body,{'Content-Type':'application/json'})
        response=conn.getresponse()
        if response.status!=200:raise ValueError('endpoint refused evidence query')
        raw=response.read(16*1024*1024)
        return json.loads(raw)['scores']
    finally:conn.close()
