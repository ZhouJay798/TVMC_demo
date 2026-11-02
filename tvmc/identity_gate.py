"""Five preregistered custodians and session-bound Schnorr admission.

Public commitments are pseudonymous identifiers; external authentication,
secure channels, durable replay protection and trust in the registry are not
implemented. This is an experimental Fiat-Shamir proof, not a production IdP.
"""
from dataclasses import dataclass
from .crypto import G,P,Q,scalar,digest


def _challenge(domain,custodian,commitment,announcement,context):
    return int(digest([domain,custodian,commitment,announcement,context]),16)%Q


def identity_proof(custodian,secret,context):
    if type(custodian) is not int or not 1<=custodian<=5 or not 0<secret<Q:
        raise ValueError('invalid identity')
    commitment=pow(G,secret,P);randomizer=scalar();announcement=pow(G,randomizer,P)
    challenge=_challenge('TVMC/identity-proof/v1',custodian,commitment,announcement,context)
    return {'i':custodian,'a':announcement,'z':(randomizer+challenge*secret)%Q}


def verify_identity(commitment,proof,context):
    try:
        i,a,z=proof['i'],proof['a'],proof['z']
        if type(i) is not int or not 1<=i<=5 or type(a) is not int or type(z) is not int:
            return False
        if not (1<commitment<P and pow(commitment,Q,P)==1 and 1<a<P and pow(a,Q,P)==1 and 0<=z<Q):
            return False
        c=_challenge('TVMC/identity-proof/v1',i,commitment,a,context)
        return pow(G,z,P)==a*pow(commitment,c,P)%P
    except (KeyError,TypeError,ValueError):return False


def enrollment_context(i):return {'domain':'TVMC/custodian-enrollment/v1','custodian':i}


@dataclass(frozen=True)
class Admission:
    sid:str
    epoch:int
    participants:tuple
    base_sha256:str
    nonce:str
    identity_commitments:tuple
    seed:str
    proofs:tuple


class IdentityGate:
    def __init__(self,records):
        self._registry={};self._sessions={}
        for record in records:
            i=record['i'];commitment=record['commitment'];proof=record['proof']
            if i in self._registry or proof.get('i')!=i or not verify_identity(commitment,proof,enrollment_context(i)):
                raise ValueError('invalid or duplicate preregistration')
            self._registry[i]=commitment
        if set(self._registry)!=set(range(1,6)) or len(set(self._registry.values()))!=5:
            raise ValueError('exactly five distinct preregistered identities required')

    @property
    def registry(self):return dict(self._registry)

    @property
    def sessions(self):return dict(self._sessions)

    def require_admitted(self, admission):
        if not isinstance(admission, Admission) or self._sessions.get(admission.sid) is not admission:
            raise ValueError('session has not been admitted by this gate')
        return admission

    @staticmethod
    def context(sid,epoch,participants,base_sha256,nonce):
        return {'domain':'TVMC/session-admission/v1','sid':sid,'epoch':epoch,
                'participants':list(participants),'base_sha256':base_sha256,'nonce':nonce}

    def register(self,sid,epoch,participants,base_sha256,nonce,proofs):
        ids=list(participants)
        if (not isinstance(sid,str) or not sid or sid in self._sessions or type(epoch) is not int or epoch<0 or
            not isinstance(nonce,str) or not nonce or not isinstance(base_sha256,str) or len(base_sha256)!=64 or
            len(ids)<2 or len(ids)>5 or any(type(i) is not int or i not in self._registry for i in ids) or
            len(set(ids))!=len(ids)):
            raise ValueError('invalid session request')
        ids=sorted(ids);ctx=self.context(sid,epoch,ids,base_sha256,nonce)
        if (len(proofs)!=len(ids) or {p.get('i') for p in proofs}!=set(ids) or
            len({p.get('i') for p in proofs})!=len(proofs) or
            not all(verify_identity(self._registry[p['i']],p,ctx) for p in proofs)):
            raise ValueError('admission proofs do not match preregistered participants')
        commitments=tuple((i,self._registry[i]) for i in ids)
        seed=digest(['TVMC/session-fingerprint/v1',sid,epoch,commitments])
        result=Admission(sid,epoch,tuple(ids),base_sha256,nonce,commitments,seed,tuple(dict(p) for p in proofs))
        self._sessions[sid]=result
        return result
