from dataclasses import replace
import copy
import pytest
from tvmc.epochs import create,contribution,refresh,BoundShare,rotate_session_identity
from tvmc.crypto import scalar,verify_proof,G,P,Q

@pytest.mark.parametrize('ids',[(1,2),(2,3,4)])
def test_refresh_and_identity_are_independent(ids):
    identities=[scalar() for _ in range(4)]
    key,old=create('session',0,ids,identities);proofs=old.proofs()
    current=refresh(old,[contribution(old,i) for i in ids])
    assert current.public.identities==old.public.identities
    assert current.restore(current.shares,current.proofs())==key
    assert old.restore(old.shares,proofs)==key
    with pytest.raises(ValueError):current.restore(old.shares,proofs)
    forged=[replace(s,epoch=1) for s in old.shares]
    with pytest.raises(ValueError):current.restore(forged,proofs)
    rotated=rotate_session_identity(current,[scalar() for _ in range(4)])
    assert rotated.public.commitments==current.public.commitments
    assert all(verify_proof(old.public,p,old.context()) for p in proofs)
    assert not any(verify_proof(rotated.public,p,old.context()) for p in proofs)

@pytest.mark.parametrize('fault',['constant','update','missing','duplicate','context'])
def test_rejected_round_is_atomic(fault):
    key,old=create('session',0,(2,3,4),[scalar() for _ in range(4)])
    contributions=[contribution(old,i) for i in (2,3,4)];before=copy.deepcopy(old)
    if fault=='constant':contributions[0]['commitments']=(G,)+contributions[0]['commitments'][1:]
    if fault=='update':contributions[0]['updates'][2]=(contributions[0]['updates'][2]+1)%Q
    if fault=='missing':contributions.pop()
    if fault=='duplicate':contributions[1]=contributions[0]
    if fault=='context':contributions[0]['sid']='other'
    with pytest.raises(ValueError):refresh(old,contributions)
    assert old==before
