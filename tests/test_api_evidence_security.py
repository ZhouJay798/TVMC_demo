import copy
import math
import pytest
from unittest.mock import patch
from tvmc.paper_workflow import Custody,sign
from tvmc.evidence import collect_quorum,verify_quorum,read_scores
from tvmc.crypto import digest

@pytest.fixture(scope='module')
def fixture():
    c=Custody();target=dict(endpoint='https://fixture.invalid/infer',tls_sha256='00'*32,start=100.,end=105.)
    probes=[dict(text='fixture text',tokens=[1,2],bit=0)]*3
    records=[]
    for i in (1,2,3):
        b=dict(member=i,group='fixture',target=target,probe_hash=digest(probes),scores=[2.,2.,2.],time=101.,finished=102.,evidence_version=2)
        records.append(dict(body=b,signature=sign(c.members[i-1].identity,'api-evidence',b)))
    return c,target,probes,records

@pytest.mark.parametrize('attack',['endpoint','tls','window','group','probe_text','probe_token','short_quorum','duplicate','identity','signature','raw_score','late_start','late_finish','finish_before_start','missing_finish','nan_finish','version','mixed_versions','short_response','nan_score','infinite_score','aggregate_overflow'])
def test_reject_invalid_evidence(fixture,attack):
    c,target,probes,records=fixture
    target=copy.deepcopy(target);probes=copy.deepcopy(probes);records=copy.deepcopy(records);group='fixture'
    b=records[0]['body'];resign=False
    if attack=='endpoint':target['endpoint']='https://other.invalid/infer'
    elif attack=='tls':target['tls_sha256']='11'*32
    elif attack=='window':target['end']=106.
    elif attack=='group':group='new-investigation'
    elif attack=='probe_text':probes[0]['text']='different'
    elif attack=='probe_token':probes[0]['tokens']=[2,3]
    elif attack=='short_quorum':records=records[:2]
    elif attack=='duplicate':records=[records[0]]*3
    elif attack=='identity':b['member']=4
    elif attack=='signature':records[0]['signature']='00'*64
    elif attack=='raw_score':b['scores']=[-2.,-2.,-2.]
    elif attack=='late_start':b['time']=106.;resign=True
    elif attack=='late_finish':b['finished']=106.;resign=True
    elif attack=='finish_before_start':b['finished']=100.;resign=True
    elif attack=='missing_finish':del b['finished'];resign=True
    elif attack=='nan_finish':b['finished']=float('nan');resign=True
    elif attack=='version':b['evidence_version']=99;resign=True
    elif attack=='mixed_versions':del b['evidence_version'];del b['finished'];resign=True
    elif attack=='short_response':b['scores']=[2.];resign=True
    elif attack=='nan_score':b['scores']=[float('nan')]*3;resign=True
    elif attack=='infinite_score':b['scores']=[float('inf')]*3;resign=True
    elif attack=='aggregate_overflow':b['scores']=[1e308]*3;resign=True
    if resign:records[0]['signature']=sign(c.members[0].identity,'api-evidence',b)
    with pytest.raises(ValueError):verify_quorum(records,c.registry,3,target,probes,group)


def test_normal_and_legacy_are_distinguished(fixture):
    c,target,probes,records=fixture
    assert verify_quorum(records,c.registry,3,target,probes,'fixture')==[1]
    legacy=copy.deepcopy(records)
    for r in legacy:
        del r['body']['finished'];del r['body']['evidence_version']
        r['signature']=sign(c.members[r['body']['member']-1].identity,'api-evidence',r['body'])
    assert verify_quorum(legacy,c.registry,3,target,probes,'fixture')==[1]


def test_collector_rejects_late_response_before_logging(fixture):
    c,target,probes,_=fixture;group='late-response'
    c.ledger.append('endpoint_commitment',dict(group=group,target=target))
    clock=[102.];durations=iter([1.,1.,2.])
    def query(*args):clock[0]+=next(durations);return [2.,2.,2.]
    with patch('tvmc.evidence.time.time',side_effect=lambda:clock[0]):
        with pytest.raises(ValueError,match='response completed'):
            collect_quorum(c,target,probes,query,[1,2,3],group)
    assert not any(e['kind']=='endpoint_evidence' and e['body']['group']==group for e in c.ledger.entries)


@pytest.mark.parametrize('repeats,delta',[(0,1),(-1,1),(True,1),(3,0),(3,float('nan')),(3,float('inf'))])
def test_invalid_readout_settings(repeats,delta):
    with pytest.raises(ValueError):read_scores([2.,2.,2.],repeats,delta)


def test_nondefault_probe_repeats_reach_public_judge(fixture):
    import time
    from tvmc.paper_trace import FileTracer,escrow_parameters,disclose_parameters,judge_endpoint
    from tvmc.crypto import canonical
    from tvmc.paper_workflow import sha
    c=fixture[0];group='two-repeats';repeats=2
    probes=[dict(text='repeat fixture',tokens=[1,2],bit=k) for k in range(4) for _ in range(repeats)]
    spec=dict(kind='signed-api-logit-difference',probe_hash=digest(probes),repeats=repeats,delta=1.)
    tracer=FileTracer(c,[.5]*4,2,group,reader_spec=spec)
    cap=escrow_parameters(c,group,dict(probes=probes,repeats=repeats,delta=1.))
    tracer.enroll('two-repeat-copy',[1,0,1,0]);c.issue(1,b'model-fixture','two-repeat-copy')
    t=time.time();target=dict(endpoint='https://repeat.invalid',tls_sha256='fixture',start=t-1,end=t+120)
    c.ledger.append('endpoint_commitment',dict(group=group,target=target));artifact=canonical(target);tracer.commit_file(artifact)
    disclose_parameters(c,group,cap,[2,3,4],sha(artifact))
    word,records=collect_quorum(c,target,probes,lambda *args:[2.,2.,-2.,-2.,2.,2.,-2.,-2.],[2,3,4],group,repeats)
    reader=lambda _:verify_quorum(records,c.registry,3,target,probes,group,repeats)
    result=tracer.trace_file(artifact,reader,[2,3,4])
    assert result['recipients']==[1]
    assert judge_endpoint(result,target,probes,records,c,c.ledger.entries[-1]['hash'])


def test_failed_collection_cannot_reuse_group(fixture):
    c,target,probes,_=fixture;group='failed-collection'
    c.ledger.append('endpoint_commitment',dict(group=group,target=target))
    calls=[]
    def failing(*args):calls.append(1);raise ValueError('transport failed')
    with patch('tvmc.evidence.time.time',return_value=102.):
        with pytest.raises(ValueError,match='transport failed'):collect_quorum(c,target,probes,failing,[1,2,3],group)
        with pytest.raises(ValueError,match='consumed'):collect_quorum(c,target,probes,failing,[1,2,3],group)
    assert len(calls)==1


def test_uncommitted_target_never_queried(fixture):
    c,target,probes,_=fixture
    def forbidden(*args):pytest.fail('queried uncommitted endpoint')
    with pytest.raises(ValueError,match='commitment missing'):
        collect_quorum(c,target,probes,forbidden,[1,2,3],'uncommitted')
