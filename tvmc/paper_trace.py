"""Encrypted Tardos scoring, single-use file evidence, public verification.
Model readout is recomputed by the supplied actual-model reader.
"""
from .pki import issuer_key
from .pki import member_key

import math
import secrets
from .crypto import P,Q,G,scalar,digest
from . import paper_threshold as te
from .paper_workflow import sha,sign,verify


def encrypt_bit(public,bit,context):
    if bit not in (0,1): raise ValueError('not a bit')
    r=scalar(); a=pow(G,r,P); b=pow(public.pk,r,P)*pow(G,bit,P)%P
    fake=1-bit; c=[0,0]; z=[0,0]; t=[None,None]
    c[fake]=scalar();z[fake]=scalar()
    y=b*pow(pow(G,fake,P),-1,P)%P
    t[fake]=[pow(G,z[fake],P)*pow(pow(a,c[fake],P),-1,P)%P,
             pow(public.pk,z[fake],P)*pow(pow(y,c[fake],P),-1,P)%P]
    k=scalar();t[bit]=[pow(G,k,P),pow(public.pk,k,P)]
    total=te.challenge('bit',public.kid,a,b,t,context)
    c[bit]=(total-c[fake])%Q;z[bit]=(k+c[bit]*r)%Q
    return dict(a=a,b=b,c=c,z=z)


def verify_bit(public,ct,context):
    try:
        a,b,c,z=(ct[k] for k in ('a','b','c','z'))
        if not te.point(a) or not te.point(b) or len(c)!=2 or len(z)!=2: return False
        if any(type(v) is not int or not 0<=v<Q for v in c+z): return False
        t=[]
        for bit in (0,1):
            y=b*pow(pow(G,bit,P),-1,P)%P
            t.append([pow(G,z[bit],P)*pow(pow(a,c[bit],P),-1,P)%P,
                      pow(public.pk,z[bit],P)*pow(pow(y,c[bit],P),-1,P)%P])
        return sum(c)%Q==te.challenge('bit',public.kid,a,b,t,context)
    except (KeyError,ValueError,TypeError): return False


def discrete_log_signed(element,bound):
    """BSGS over a bounded signed integer range, not over the private-key space."""
    m=math.isqrt(2*bound+1)+1
    table={}; cur=1
    for j in range(m): table.setdefault(cur,j);cur=cur*G%P
    shifted=element*pow(G,bound,P)%P; factor=pow(pow(G,m,P),-1,P)
    for i in range(m+1):
        if shifted in table:
            result=i*m+table[shifted]-bound
            if -bound<=result<=bound: return result
        shifted=shifted*factor%P
    raise ValueError('score outside declared range')


class FileTracer:
    def __init__(self,custody,biases,threshold,group='group-1',scale=1024,reader_spec=None):
        self.custody=custody;self.biases=list(map(float,biases));self.threshold=float(threshold)
        if not self.biases or any(not 0<p<1 for p in self.biases):raise ValueError('bad biases')
        self.group=group;self.scale=scale;self.codebook={};self.retired=False;self.evidence=None
        self.params=dict(group=group,biases=self.biases,threshold=self.threshold,scale=scale,
                         reader=reader_spec or {'kind':'synthetic-fixture'})
        self.salt=secrets.token_hex(32)
        self.param_seq=custody.ledger.append('parameters_commitment',dict(group=group,commitment=digest([self.params,self.salt])))

    def enroll(self,cid,word):
        if self.evidence is not None or self.retired:raise ValueError('issuance closed')
        if cid in self.codebook or len(word)!=len(self.biases):raise ValueError('invalid enrollment')
        ct=[encrypt_bit(self.custody.code_key,int(w),[self.group,cid,k]) for k,w in enumerate(word)]
        if not all(verify_bit(self.custody.code_key,c,[self.group,cid,k]) for k,c in enumerate(ct)):
            raise ValueError('invalid encrypted bit')
        self.codebook[cid]=ct
        self.custody.ledger.append('encrypted_codeword',dict(group=self.group,cid=cid,ciphertexts=ct))

    def commit_file(self,artifact):
        if self.evidence is not None or self.retired:raise ValueError('group already reserved')
        self.evidence=sha(artifact)
        self.custody.ledger.append('evidence_commitment',dict(group=self.group,file_sha256=self.evidence))

    def trace_file(self,artifact,reader,indices):
        if self.retired or self.evidence is None or sha(artifact)!=self.evidence:
            raise ValueError('missing commitment, substituted file or spent group')
        pub=self.custody.code_key
        if len(set(indices))!=len(indices) or len(indices)<pub.t or any(i not in self.custody.registry for i in indices):
            raise ValueError('need distinct registered score approvers')
        # Burn on disclosure, including failures, so retries cannot obtain fresh linear equations.
        self.retired=True
        self.custody.ledger.append('group_disclosed_and_retired',dict(parameters=self.params,salt=self.salt))
        observed=list(reader(artifact))
        if len(observed)!=len(self.biases) or any(v not in (-1,0,1) for v in observed):
            raise ValueError('unreadable observation')
        A=[];B=[]
        for p,y in zip(self.biases,observed):
            u=math.sqrt((1-p)/p);v=math.sqrt(p/(1-p))
            A.append(0 if y==-1 else (u+v)*(1 if y==1 else -1))
            B.append(0 if y==-1 else (-v if y==1 else v))
        ai=[round(self.scale*a) for a in A];bi=round(self.scale*sum(B))
        cutoff=math.ceil(self.scale*self.threshold+(len(ai)+1)/2)
        bound=sum(abs(a) for a in ai)+abs(bi)+1
        from .score_approval import ScoreApprover, score_ciphertext
        import copy
        head=self.custody.ledger.entries[-1]['hash']
        request=dict(group=self.group,kid=pub.kid,head=head,evidence=self.evidence,
                     observed=observed,scores=[dict(cid=cid,**score_ciphertext(self.params,observed,cts))
                                               for cid,cts in self.codebook.items()])
        self.custody.ledger.append('score_request',request)
        snapshot=copy.deepcopy(self.custody.ledger.entries)
        head=snapshot[-1]['hash']
        batches={}
        for i in indices:
            member=self.custody.members[i-1]
            approver=ScoreApprover(pub,self.custody.code_shares[i-1],member.identity,
                                  self.custody.issuer_pk,self.custody.registry,reader)
            batches[i]=approver.approve(request,snapshot,head,artifact)
        for batch in batches.values():
            self.custody.ledger.append('score_approval',dict(approval=batch['approval'],
                signature=batch['approval_signature']))
        scores={};hits=[];records=[]
        for cid,cts in self.codebook.items():
            # Authenticate against immutable-for-this-run enrollment, not a caller-supplied codebook.
            enrolled=[e['body'] for e in self.custody.ledger.entries if e['kind']=='encrypted_codeword' and e['body']['cid']==cid and e['body']['group']==self.group]
            if len(enrolled)!=1 or enrolled[0]['ciphertexts']!=cts:raise ValueError('codebook substitution')
            if cid not in self.custody.manifests:raise ValueError('unissued copy')
            a=1;b=pow(G,bi,P)
            for coefficient,ct in zip(ai,cts):
                a=a*pow(ct['a'],coefficient,P)%P;b=b*pow(ct['b'],coefficient,P)%P
            ctx=['score',self.group,self.evidence,observed,cid,a,b]
            proofs=[];signatures=[]
            for i in indices:
                response=next(row for row in batches[i]['responses'] if row['cid']==cid)
                pr=response['proof'];sig=response['signature']
                body=dict(context=ctx,proof=pr)
                if not verify(member_key(self.custody.registry,i),'score',body,sig):raise ValueError('score identity')
                proofs.append(pr);signatures.append(sig)
            mask=te.combine(pub,a,proofs,ctx)
            score=discrete_log_signed(b*pow(mask,-1,P)%P,bound)
            scores[cid]=score/self.scale
            if score>=cutoff:hits.append(cid)
            records.append(dict(cid=cid,a=a,b=b,proofs=proofs,signatures=signatures,integer_score=score,context=ctx))
        result=dict(protocol="member-checked-scores-v1",request_hash=digest(request),group=self.group,file_sha256=self.evidence,observed=observed,scores=scores,
                    cutoff_integer=cutoff,hits=hits,recipients=[self.custody.manifests[c]['message']['recipient'] for c in hits],
                    evidence=records,scope=self.params['reader'])
        self.custody.ledger.append('trace_result',result)
        return result


def judge_file(result,artifact,reader,entries,expected_head,issuer_pk,registry,public):
    """Public recomputation using caller-pinned roots. No shares or private keys.

    Hash chain integrity alone is NOT an independent transparency witness.
    """
    from .paper_workflow import Ledger
    try:
        issuer_pk=issuer_key(registry,issuer_pk)
        ledger=Ledger();ledger.entries=entries
        if not ledger.check(expected_head) or result['file_sha256']!=sha(artifact):return False
        group=result['group']; observed=list(reader(artifact))
        if observed!=result['observed']:return False
        disclosure=[e for e in entries if e['kind']=='group_disclosed_and_retired' and e['body']['parameters']['group']==group]
        commitments=[e for e in entries if e['kind']=='parameters_commitment' and e['body']['group']==group]
        evidence=[e for e in entries if e['kind']=='evidence_commitment' and e['body']['group']==group]
        traces=[e for e in entries if e['kind']=='trace_result' and e['body']['group']==group]
        if any(len(x)!=1 for x in (disclosure,commitments,evidence,traces)):return False
        if traces[0]['body']!=result or not evidence[0]['seq']<disclosure[0]['seq']<traces[0]['seq']:return False
        if evidence[0]['body']['file_sha256']!=sha(artifact):return False
        params=disclosure[0]['body']['parameters'];salt=disclosure[0]['body']['salt']
        if commitments[0]['body']['commitment']!=digest([params,salt]):return False
        biases=params['biases'];scale=params['scale']
        if len(observed)!=len(biases) or any(y not in (-1,0,1) for y in observed):return False
        A=[];B=[]
        for p,y in zip(biases,observed):
            u=math.sqrt((1-p)/p);v=math.sqrt(p/(1-p))
            A.append(0 if y==-1 else (u+v)*(1 if y==1 else -1))
            B.append(0 if y==-1 else (-v if y==1 else v))
        ai=[round(scale*a) for a in A];bi=round(scale*sum(B))
        cutoff=math.ceil(scale*params['threshold']+(len(ai)+1)/2)
        if result['cutoff_integer']!=cutoff:return False
        bound=sum(abs(a) for a in ai)+abs(bi)+1
        codes=[e for e in entries if e['kind']=='encrypted_codeword' and e['body']['group']==group]
        if len(result['evidence'])!=len(codes):return False
        if result.get('protocol') == 'member-checked-scores-v1':
            requests=[e for e in entries if e['kind']=='score_request' and e['body']['group']==group]
            if len(requests)!=1:return False
            request=requests[0];req=request['body']
            if (digest(req)!=result['request_hash'] or req['kid']!=public.kid or
                req['observed']!=observed or req['evidence']!=sha(artifact) or
                not disclosure[0]['seq']<request['seq']<traces[0]['seq'] or
                req['head']!=request['prev']):return False
            expected=[dict(cid=r['cid'],a=r['a'],b=r['b']) for r in result['evidence']]
            if req['scores']!=expected:return False
            approved=set()
            for entry in entries:
                if entry['kind']!='score_approval':continue
                body=entry['body'];approval=body['approval']
                if approval['request_hash']!=result['request_hash']:continue
                i=approval['member']
                if (i in approved or i not in registry or
                    not request['seq']<entry['seq']<traces[0]['seq'] or
                    not verify(member_key(registry,i),'score-approval',approval,body['signature'])):return False
                approved.add(i)
            if len(approved)<public.t:return False
            if any(p['i'] not in approved for r in result['evidence'] for p in r['proofs']):return False
        records={x['cid']:x for x in result['evidence']}
        if len(records)!=len(codes):return False
        scores={};hits=[];recipients=[]
        for code in codes:
            cid=code['body']['cid'];cts=code['body']['ciphertexts'];record=records[cid]
            manifests=[e for e in entries if e['kind']=='manifest' and e['body']['message']['cid']==cid]
            if len(manifests)!=1:return False
            man=manifests[0]['body'];msg=man['message']
            member_key(registry,msg['recipient'])
            member_key(registry,msg['recipient'],'receiving')
            if not commitments[0]['seq']<code['seq']<manifests[0]['seq']<evidence[0]['seq']:return False
            if not verify(issuer_pk,'manifest',msg,man['signature']):return False
            all_codes=[e['body'] for e in entries if e['kind']=='encrypted_codeword' and e['body']['cid']==cid]
            if msg['codebook_hash']!=digest(all_codes) or msg['registration']!=registry.get(msg['recipient']):return False
            if len(cts)!=len(ai):return False
            a=1;b=pow(G,bi,P)
            for k,(coefficient,ct) in enumerate(zip(ai,cts)):
                if not verify_bit(public,ct,[group,cid,k]):return False
                a=a*pow(ct['a'],coefficient,P)%P;b=b*pow(ct['b'],coefficient,P)%P
            context=['score',group,sha(artifact),observed,cid,a,b]
            if context!=record['context'] or a!=record['a'] or b!=record['b']:return False
            proofs=record['proofs'];sigs=record['signatures']
            if len(proofs)!=len(sigs):return False
            for proof,sig in zip(proofs,sigs):
                if not verify(member_key(registry,proof['i']),'score',dict(context=context,proof=proof),sig):return False
            mask=te.combine(public,a,proofs,context)
            score=discrete_log_signed(b*pow(mask,-1,P)%P,bound)
            if record['integer_score']!=score:return False
            scores[cid]=score/scale
            if score>=cutoff:hits.append(cid);recipients.append(msg['recipient'])
        return scores==result['scores'] and hits==result['hits'] and recipients==result['recipients']
    except (KeyError,ValueError,TypeError,OverflowError,ZeroDivisionError):
        return False


def escrow_parameters(custody,group,parameters):
    """Must be called before any release. Only ciphertext and commitment are logged."""
    from .crypto import canonical
    salt=secrets.token_hex(32);payload=canonical(dict(parameters=parameters,salt=salt))
    capsule=te.seal(custody.release_key,payload,['probes',group])
    custody.ledger.append('probe_escrow',dict(group=group,commitment=digest([parameters,salt]),capsule=capsule))
    return capsule


def disclose_parameters(custody,group,capsule,indices,evidence_hash):
    """Quorum-signed probe release tied to a pre-existing investigation commitment."""
    import json
    if len(set(indices))!=len(indices) or len(indices)<custody.release_key.t:raise ValueError('probe quorum missing')
    entries=custody.ledger.entries
    if not any(e['kind']=='evidence_commitment' and e['body'].get('group')==group and e['body'].get('file_sha256')==evidence_hash for e in entries):
        raise ValueError('evidence must precede disclosure')
    if any(e['kind']=='probe_disclosure' and e['body']['group']==group for e in entries):raise ValueError('probe group already disclosed')
    if not te.check_capsule(custody.release_key,capsule,['probes',group]):raise ValueError('invalid investigation capsule')
    ctx=['investigation',group,evidence_hash,digest(capsule)];proofs=[];signed=[]
    for i in indices:
        if i not in custody.registry:raise ValueError('unknown member')
        pr=te.partial(custody.release_key,custody.release_shares[i-1],capsule['a'],ctx)
        proofs.append(pr);signed.append(sign(custody.members[i-1].identity,'investigation',dict(context=ctx,proof=pr)))
    payload=te.open_capsule(custody.release_key,capsule,proofs,['probes',group],ctx)
    decoded=json.loads(payload)
    escrow=[e for e in entries if e['kind']=='probe_escrow' and e['body']['group']==group]
    if len(escrow)!=1 or escrow[0]['body']['commitment']!=digest([decoded['parameters'],decoded['salt']]):raise ValueError('probe commitment mismatch')
    custody.ledger.append('probe_disclosure',dict(group=group,evidence_hash=evidence_hash,proofs=proofs,signatures=signed,opening=decoded))
    return decoded['parameters']


def judge_endpoint(result,target,probes,records,custody,expected_head):
    """Public endpoint checks before the common encrypted-score verifier."""
    from .crypto import canonical
    from .evidence import verify_quorum
    group=result['group'];entries=custody.ledger.entries
    try:
        commitments=[e for e in entries if e['kind']=='endpoint_commitment' and e['body']==dict(group=group,target=target)]
        collected=[e for e in entries if e['kind']=='endpoint_evidence' and e['body']['group']==group]
        disclosures=[e for e in entries if e['kind']=='probe_disclosure' and e['body']['group']==group]
        if any(len(x)!=1 for x in (commitments,collected,disclosures)):return False
        if not commitments[0]['seq']<disclosures[0]['seq']<collected[0]['seq']:return False
        if collected[0]['body']['records']!=records:return False
        if any(e['kind']=='legitimate_endpoint' and e['body']['endpoint']==target['endpoint'] for e in entries):return False
        openings=[e['body']['parameters'] for e in entries if e['kind']=='group_disclosed_and_retired' and e['body']['parameters']['group']==group]
        if len(openings)!=1:return False
        repeats=openings[0]['reader'].get('repeats',3)
        reader=lambda _:verify_quorum(records,custody.registry,custody.release_key.t,target,probes,group,repeats)
        return judge_file(result,canonical(target),reader,entries,expected_head,custody.issuer_pk,custody.registry,custody.code_key)
    except (KeyError,ValueError,TypeError):return False


def rescore_rebuttal(custody,tracer,result,cid,erased,indices,rebuttal_digest):
    """Authorized recomputation after verified rebuttal; no new arbitrary word.

    Caller must validate anchored log first. The allowed request is constrained to
    deleting positions from the already adjudicated word for this one recipient.
    """
    if not tracer.retired or result['group']!=tracer.group or cid not in result['hits']:raise ValueError('no accusation to rebut')
    valid=[e for e in custody.ledger.entries if e['kind']=='verified_rebuttal' and e['body']==dict(group=tracer.group,cid=cid,erased=sorted(set(erased)),digest=rebuttal_digest)]
    if len(valid)!=1:raise ValueError('verified rebuttal required')
    if len(set(indices))!=len(indices) or len(indices)<custody.code_key.t:raise ValueError('score quorum')
    word=list(result['observed'])
    for k in erased:
        if not 0<=k<len(word):raise ValueError('invalid erasure position')
        word[k]=-1
    ai=[];B=[]
    for p,y in zip(tracer.biases,word):
        u=math.sqrt((1-p)/p);v=math.sqrt(p/(1-p))
        ai.append(round(tracer.scale*(0 if y==-1 else (u+v)*(1 if y==1 else -1))))
        B.append(0 if y==-1 else (-v if y==1 else v))
    bi=round(tracer.scale*sum(B));a=1;b=pow(G,bi,P)
    for coeff,ct in zip(ai,tracer.codebook[cid]):a=a*pow(ct['a'],coeff,P)%P;b=b*pow(ct['b'],coeff,P)%P
    ctx=['rebuttal-score',digest(result),cid,sorted(set(erased)),rebuttal_digest,a,b]
    proofs=[te.partial(custody.code_key,custody.code_shares[i-1],a,ctx) for i in indices]
    signatures=[sign(custody.members[i-1].identity,'rebuttal-score',dict(context=ctx,proof=p)) for i,p in zip(indices,proofs)]
    mask=te.combine(custody.code_key,a,proofs,ctx)
    value=discrete_log_signed(b*pow(mask,-1,P)%P,sum(abs(v) for v in ai)+abs(bi)+1)
    out=dict(cid=cid,score=value/tracer.scale,retained=value>=result['cutoff_integer'],relation='relay',
             erased=sorted(set(erased)),proofs=proofs,signatures=signatures,context=ctx)
    custody.ledger.append('rebuttal_result',out);return out


def judge_rebuttal(original,rebuttal_bundle,positions,entries,registry,public):
    """Verify the allowed erasure rescore using only public material."""
    try:
        out=rebuttal_bundle['result'];cid=out['cid'];group=original['group']
        if cid not in original['hits'] or out['erased']!=sorted(set(positions)):return False
        opening=next(e['body']['parameters'] for e in entries if e['kind']=='group_disclosed_and_retired' and e['body']['parameters']['group']==group)
        cts=next(e['body']['ciphertexts'] for e in entries if e['kind']=='encrypted_codeword' and e['body']['cid']==cid and e['body']['group']==group)
        word=list(original['observed']);scale=opening['scale'];A=[];B=[]
        for k in positions:word[k]=-1
        for p,y in zip(opening['biases'],word):
            u=math.sqrt((1-p)/p);v=math.sqrt(p/(1-p))
            A.append(round(scale*(0 if y==-1 else (u+v)*(1 if y==1 else -1))))
            B.append(0 if y==-1 else (-v if y==1 else v))
        bi=round(scale*sum(B));a=1;b=pow(G,bi,P)
        for coeff,ct in zip(A,cts):a=a*pow(ct['a'],coeff,P)%P;b=b*pow(ct['b'],coeff,P)%P
        rdigest=digest(rebuttal_bundle['rebuttal'])
        context=['rebuttal-score',digest(original),cid,sorted(set(positions)),rdigest,a,b]
        if out['context']!=context or len(out['proofs'])!=len(out['signatures']):return False
        for proof,sig in zip(out['proofs'],out['signatures']):
            if not verify(member_key(registry,proof['i']),'rebuttal-score',dict(context=context,proof=proof),sig):return False
        shared=te.combine(public,a,out['proofs'],context)
        value=round(out['score']*scale)
        if abs(value)>sum(abs(v) for v in A)+abs(bi)+1:return False
        if b*pow(shared,-1,P)%P!=pow(G,value,P):return False
        return out['retained']==(value>=original['cutoff_integer']) and out['relation']=='relay'
    except (KeyError,ValueError,TypeError,IndexError):return False
