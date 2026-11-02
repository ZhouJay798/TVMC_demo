"""Session-level leads from decoded sessions and a caller-trusted registry snapshot.

This function performs set operations only. It does not authenticate a registry,
validate a fingerprint receipt, make individual accusations, or run Judge.
Callers must validate those inputs according to their evidence workflow.
"""

def session_leads(decoded_sessions, records):
    if not isinstance(decoded_sessions, (list, tuple)):
        raise ValueError('decoded sessions must be a sequence')
    registry={}
    for record in records:
        if not isinstance(record,dict):
            raise ValueError('invalid registry record')
        sid=record.get('sid');participants=record.get('participants');epoch=record.get('epoch')
        if not isinstance(sid,str) or not sid or sid in registry:
            raise ValueError('invalid or duplicate session id')
        if type(epoch) is not int or epoch<0:
            raise ValueError('invalid epoch')
        if (not isinstance(participants,(list,tuple)) or not participants or
            any(type(i) is not int or i<1 for i in participants) or len(set(participants))!=len(participants)):
            raise ValueError('invalid participants')
        registry[sid]={'sid':sid,'epoch':epoch,'participants':sorted(participants)}
    if any(not isinstance(sid,str) or not sid for sid in decoded_sessions) or len(set(decoded_sessions))!=len(decoded_sessions):
        raise ValueError('invalid or duplicate decoded session')
    if any(sid not in registry for sid in decoded_sessions):
        raise ValueError('decoded session missing from trusted snapshot')
    selected=[registry[sid] for sid in decoded_sessions]
    if not selected:
        return {'classification':'无会话命中','sessions':[],'intersection':None,'union':[],
                'suggested_leads':[],'rule':'no_hits','individual_attribution':False}
    sets=[set(r['participants']) for r in selected]
    common=sorted(set.intersection(*sets));union=sorted(set.union(*sets))
    if len(selected)==1:
        rule='single_session';leads=union
    elif common:
        rule='intersection_heuristic';leads=common
    else:
        rule='union_fallback';leads=union
    return {'classification':'会话级线索','sessions':selected,'intersection':common,'union':union,
            'suggested_leads':leads,'rule':rule,'individual_attribution':False,
            'caveat':'交集是启发式线索，登记参与方不等于泄漏者；需要独立证据，不作个人归责。'}
