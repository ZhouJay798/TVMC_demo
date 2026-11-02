import pytest
from tvmc.cross_session import session_leads

RECORDS=[{'sid':'A-E0','epoch':0,'participants':[1,2]},
         {'sid':'B-E0','epoch':0,'participants':[2,3,4]},
         {'sid':'C-E1','epoch':1,'participants':[5,6]}]

def test_overlapping_sessions_are_heuristic_not_attribution():
    r=session_leads(['A-E0','B-E0'],RECORDS)
    assert r['sessions']==RECORDS[:2]
    assert r['intersection']==[2] and r['union']==[1,2,3,4]
    assert r['suggested_leads']==[2] and r['rule']=='intersection_heuristic'
    assert r['classification']=='会话级线索' and r['individual_attribution'] is False

def test_disjoint_falls_back_to_union():
    r=session_leads(['A-E0','C-E1'],RECORDS)
    assert r['intersection']==[] and r['union']==[1,2,5,6]
    assert r['suggested_leads']==[1,2,5,6] and r['rule']=='union_fallback'

def test_no_hits_and_single_are_distinct():
    n=session_leads([],RECORDS)
    assert n['intersection'] is None and n['suggested_leads']==[] and n['classification']=='无会话命中'
    s=session_leads(['A-E0'],RECORDS)
    assert s['suggested_leads']==[1,2] and s['rule']=='single_session' and not s['individual_attribution']

@pytest.mark.parametrize('decoded,records',[
    (['missing'],RECORDS),(['A-E0','A-E0'],RECORDS),
    (['A-E0'],RECORDS+[RECORDS[0]]),
    (['A-E0'],[{'sid':'A-E0','epoch':0,'participants':[1,1]}]),
    (['A-E0'],[{'sid':'A-E0','epoch':0,'participants':[]}]),
    (['A-E0'],[{'sid':'A-E0','epoch':0,'participants':[True]}]),
    (['A-E0'],[{'sid':'A-E0','epoch':-1,'participants':[1]}]),
])
def test_invalid_data_fails_closed(decoded,records):
    with pytest.raises(ValueError):session_leads(decoded,records)
