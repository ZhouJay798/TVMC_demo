"""Experimental unanimous custody, atomic zero-polynomial refresh and epoch labels.
Identity rotation is external to share refresh. Historical public snapshots are immutable.
Local prototype: no secure transport, distributed consensus or secure erasure.
"""
from dataclasses import dataclass, replace
from .crypto import P,Q,G,Public,Share,scalar,prove,verify_proof,recover,digest

@dataclass(frozen=True)
class EpochPublic(Public):
    participants: tuple = ()
    def __post_init__(self):
        if (not 2 <= len(self.participants) <= self.n or
            len(set(self.participants)) != len(self.participants) or
            any(type(i) is not int or not 1 <= i <= self.n for i in self.participants) or
            self.threshold != len(self.participants) or
            len(self.commitments) != self.threshold or len(self.identities) != self.n):
            raise ValueError('epoch custody requires one valid share per participant and n/n threshold')
    def y(self, i):
        if i not in self.participants:
            raise ValueError('not a registered participant')
        result = 1
        for degree, commitment in enumerate(self.commitments):
            result = result * pow(commitment, pow(i, degree, Q), P) % P
        return result
    def verify(self, share):
        return share.i in self.participants and super().verify(share)

@dataclass(frozen=True)
class BoundShare:
    sid: str
    epoch: int
    share: Share

@dataclass(frozen=True)
class Session:
    sid: str
    epoch: int
    public: EpochPublic
    shares: tuple
    def context(self):
        return {'domain':'TVMC/epoch-custody/v1','sid':self.sid,'share_epoch':self.epoch,
                'participants':self.public.participants}
    def proofs(self):
        return [prove(self.public,s.share,self.context()) for s in self.shares]
    def restore(self, shares, proofs):
        if any(s.sid != self.sid or s.epoch != self.epoch for s in shares):
            raise ValueError('wrong session or share epoch')
        if set(s.share.i for s in shares) != set(self.public.participants):
            raise ValueError('unanimous registered membership required')
        return recover(self.public,[s.share for s in shares],proofs,self.context())


def create(sid, epoch, participants, identities):
    coeff = [scalar() for _ in participants]
    public = EpochPublic(tuple(pow(G,a,P) for a in coeff),
                         tuple(pow(G,a,P) for a in identities),len(identities),len(participants),tuple(participants))
    shares = tuple(BoundShare(sid,epoch,Share(i,sum(a*pow(i,j,Q) for j,a in enumerate(coeff))%Q,
                                             identities[i-1])) for i in participants)
    return coeff[0],Session(sid,epoch,public,shares)


def contribution(session, dealer):
    if dealer not in session.public.participants:
        raise ValueError('unknown dealer')
    coeff=[0]+[scalar() for _ in range(session.public.threshold-1)]
    return {'sid':session.sid,'from_epoch':session.epoch,'to_epoch':session.epoch+1,
            'dealer':dealer,'commitments':tuple(pow(G,a,P) for a in coeff),
            'updates':{i:sum(a*pow(i,j,Q) for j,a in enumerate(coeff))%Q for i in session.public.participants}}


def refresh(session, contributions):
    """Validate the entire round before returning a new immutable Session."""
    ids=session.public.participants
    if len(contributions)!=len(ids) or {c['dealer'] for c in contributions}!=set(ids):
        raise ValueError('incomplete/duplicate dealers')
    for c in contributions:
        if (c['sid'],c['from_epoch'],c['to_epoch'])!=(session.sid,session.epoch,session.epoch+1):
            raise ValueError('wrong refresh context')
        cs=c['commitments']
        if len(cs)!=session.public.threshold or cs[0]!=1:
            raise ValueError('refresh must have zero constant')
        if any(type(v)!=int or not 1<=v<P or pow(v,Q,P)!=1 for v in cs):
            raise ValueError('invalid subgroup commitment')
        if set(c['updates'])!=set(ids):
            raise ValueError('missing update recipient')
        for i,delta in c['updates'].items():
            expected=1
            for j,C in enumerate(cs): expected=expected*pow(C,pow(i,j,Q),P)%P
            if type(delta)!=int or not 0<=delta<Q or pow(G,delta,P)!=expected:
                raise ValueError('update does not match commitments')
    cs=list(session.public.commitments)
    for c in contributions:
        cs=[a*b%P for a,b in zip(cs,c['commitments'])]
    public=replace(session.public,commitments=tuple(cs))
    shares=tuple(BoundShare(session.sid,session.epoch+1,replace(s.share,
        value=(s.share.value+sum(c['updates'][s.share.i] for c in contributions))%Q)) for s in session.shares)
    if not all(public.verify(s.share) for s in shares): raise ValueError('invalid final shares')
    return Session(session.sid,session.epoch+1,public,shares)


def rotate_session_identity(session, identities):
    return replace(session,public=replace(session.public,identities=tuple(pow(G,a,P) for a in identities)),
                   shares=tuple(replace(s,share=replace(s.share,gamma=identities[s.share.i-1])) for s in session.shares))


def interpolate(shares):
    total=0
    for s in shares:
        weight=1
        for t in shares:
            if t.i!=s.i: weight=weight*(-t.i)*pow(s.i-t.i,-1,Q)%Q
        total=(total+weight*s.value)%Q
    return total
