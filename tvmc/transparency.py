"""Persistent append-only journal with RFC6962-style Merkle roots and proofs.

Witness keys are separate role keys, but local co-location is not independent
infrastructure. Callers must retain trusted checkpoints outside this file.
"""
from .pki import issuer_key
from .pki import member_key

import hashlib,json,os
from pathlib import Path
from .crypto import canonical,digest
from .paper_workflow import Ledger,sign,verify


def leaf(data):return hashlib.sha256(b'\x00'+data).digest()
def node(a,b):return hashlib.sha256(b'\x01'+a+b).digest()


def root(leaves):
    if not leaves:return hashlib.sha256(b'').digest()
    if len(leaves)==1:return leaves[0]
    k=1<<((len(leaves)-1).bit_length()-1)
    return node(root(leaves[:k]),root(leaves[k:]))


def inclusion(leaves,i):
    if not 0<=i<len(leaves):raise ValueError('bad leaf index')
    if len(leaves)==1:return []
    k=1<<((len(leaves)-1).bit_length()-1)
    if i<k:return inclusion(leaves[:k],i)+[('right',root(leaves[k:]).hex())]
    return inclusion(leaves[k:],i-k)+[('left',root(leaves[:k]).hex())]


def verify_inclusion(entry,proof,expected):
    value=leaf(canonical(entry))
    for side,sibling in proof:
        if side not in ('left','right'):return False
        value=node(bytes.fromhex(sibling),value) if side=='left' else node(value,bytes.fromhex(sibling))
    return value.hex()==expected


class Journal(Ledger):
    def __init__(self,path):
        super().__init__();self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        if self.path.exists():
            self.entries=[json.loads(line) for line in self.path.read_text().splitlines()]
            if not self.check():raise ValueError('journal corrupt')

    def append(self,kind,body):
        # One writer in this executable. Multiwriter locks/transactions must be added
        # before exposing the journal as a concurrent service.
        seq=super().append(kind,body)
        fd=os.open(self.path,os.O_WRONLY|os.O_APPEND|os.O_CREAT,0o600)
        with os.fdopen(fd,'ab') as f:f.write(canonical(self.entries[-1])+b'\n');f.flush();os.fsync(f.fileno())
        return seq

    def checkpoint(self,signer,witnesses):
        head=dict(size=len(self.entries),root=root([leaf(canonical(e)) for e in self.entries]).hex(),chain=self.entries[-1]['hash'] if self.entries else '0'*64)
        return dict(head=head,signature=sign(signer,'tree-head',head),witnesses=[dict(i=m.i,signature=sign(m.identity,'witness',head)) for m in witnesses])


def verify_checkpoint(checkpoint,issuer_pk,registry,threshold):
    try:issuer_pk=issuer_key(registry,issuer_pk)
    except ValueError:return False
    head=checkpoint['head']
    if not verify(issuer_pk,'tree-head',head,checkpoint['signature']):return False
    ids=set()
    for w in checkpoint['witnesses']:
        i=w['i']
        if i in ids or i not in registry or not verify(member_key(registry,i),'witness',head,w['signature']):return False
        ids.add(i)
    return len(ids)>=threshold


def verify_extension(old_head,new_head,entries):
    # Full-prefix consistency verification. Correct but not logarithmic-size CT
    # consistency proofs; do not describe it as an RFC6962 compact proof.
    if old_head['size']>new_head['size'] or len(entries)!=new_head['size']:return False
    leaves=[leaf(canonical(e)) for e in entries]
    return root(leaves[:old_head['size']]).hex()==old_head['root'] and root(leaves).hex()==new_head['root']
