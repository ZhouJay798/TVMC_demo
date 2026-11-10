from contextlib import ExitStack
from dataclasses import asdict
import copy
import os
import zipfile

import pytest

from tvmc.bundle_delivery import open_bundle
from tvmc.crypto import G, P, Q, canonical, scalar, verify_proof
from tvmc.epochs import EpochPublic
from tvmc.identity_gate import IdentityGate
from tvmc.process_roles import RoleError, RoleProcess, encode_entry
from tvmc.trusted_registry import verify_issuance


def test_distinct_roles_verify_signed_records_and_own_decryption(tmp_path):
    with ExitStack() as stack:
        issuer=stack.enter_context(RoleProcess('issuer'))
        admin=stack.enter_context(RoleProcess('admin'))
        holders=[stack.enter_context(RoleProcess('custodian',custodian_id=i)) for i in range(1,6)]
        with pytest.raises(RoleError,match='duplicate replica codewords'):
            issuer.call('build_payloads',output_dir=str(tmp_path/'bad-models'),words=[[0]*256]*3,
                        replica_ids=['one','two','three'],offline_fixture=True,steps=1)
        assert issuer.call('hello')['role']=='issuer'
        pids=[issuer.info['pid'],admin.info['pid']]+[holder.info['pid'] for holder in holders]
        assert len(set(pids))==7 and os.getpid() not in pids
        assert all('private' not in str(holder.info).lower() for holder in holders)
        enrollment=[holder.call('enrollment') for holder in holders]
        gate=IdentityGate(enrollment)
        ctx=gate.context('session',0,[1,2,3],'a'*64,'nonce')
        proofs=[holder.call('identity_proof',context=ctx) for holder in holders[:3]]
        admission=gate.register('session',0,[1,2,3],'a'*64,'nonce',proofs)
        adm=asdict(admission)
        recipients=[holder.call('bind_recipient',admission=adm) for holder in holders[:3]]
        precommit=admin.call('create_precommit',enrollment=enrollment,admission=adm,
                             recipients=recipients)['precommit']
        coeff=[scalar() for _ in range(3)]
        public=EpochPublic(tuple(pow(G,a,P) for a in coeff),
                           tuple(row['commitment'] for row in enrollment),5,3,(1,2,3))
        for i,holder in enumerate(holders[:3],start=1):
            value=sum(a*pow(i,j,Q) for j,a in enumerate(coeff))%Q
            assert holder.call('store_share',value=value,public=asdict(public))['stored']
        recovery_context={'domain':'test/recovery','sid':'session','nonce':'request'}
        shares=[holder.call('prove_share',context=recovery_context) for holder in holders[:3]]
        assert all(set(s)=={'i','value','proof'} and verify_proof(public,s['proof'],recovery_context) for s in shares)
        recovered=0
        for s in shares:
            weight=1
            for other in shares:
                if s['i']!=other['i']:
                    weight=weight*(-other['i'])*pow(s['i']-other['i'],-1,Q)%Q
            recovered=(recovered+weight*s['value'])%Q
        assert recovered==coeff[0]
        payloads=[f'model-payload-{i}'.encode() for i in range(3)]
        packed=issuer.call('make_and_seal',enrollment=enrollment,admission=adm,recipients=recipients,
                           session_secret=coeff[0],payloads=[v.hex() for v in payloads])
        entries,manifest=open_bundle(gate,admission,recipients,bytes.fromhex(packed['outer']),
                                     recovered,issuer.public_key())
        assert manifest==packed['manifest']
        common={'enrollment':enrollment,'admission':adm,'recipients':recipients,'precommit':precommit,
                'manifest':manifest,'issuer_signature':packed['signature'],
                'issuer_public':issuer.info['public_key']}
        issued=admin.call('create_issuance',**common)
        verify_issuance(precommit,issued,admin.public_key(),manifest,bytes.fromhex(packed['signature']),
                        issuer.public_key(),gate,admission,recipients)
        for i,holder in enumerate(holders[:3]):
            result=holder.call('verified_open',**common,issued=issued,registry_public=admin.info['public_key'],
                               entry=encode_entry(entries[i]))
            assert result['recipient']==i+1 and result['payload_bytes']==len(payloads[i])
        with pytest.raises(RoleError):
            holders[0].call('verified_open',**common,issued=issued,registry_public=admin.info['public_key'],
                            entry=encode_entry(entries[1]))
        # An obviously wrong recipient is rejected without even attempting to
        # open a (potentially many-gigabyte) file path.
        absent_file_entry={'context':entries[1]['context'],'cipher_path':str(tmp_path/'absent.cipher'),
                           'wrapped_path':str(tmp_path/'absent.key')}
        with pytest.raises(RoleError,match='replica does not belong'):
            holders[0].call('verified_open',**common,issued=issued,registry_public=admin.info['public_key'],
                            entry=absent_file_entry)
        changed=copy.deepcopy(common);changed['manifest']['entries'][0]['context']['recipient']=2
        with pytest.raises(RoleError):admin.call('create_issuance',**changed)
        # A valid issuer signature is still insufficient when the issuer's
        # claimed recipient contradicts the admitted, precommitted mapping.
        changed['issuer_signature']=issuer.sign(canonical(changed['manifest'])).hex()
        with pytest.raises(RoleError):admin.call('create_issuance',**changed)
        with pytest.raises(RoleError):
            holders[0].call('verified_open',**changed,issued=issued,registry_public=admin.info['public_key'],
                            entry=encode_entry(entries[0]))
        outfile=tmp_path/'custodian-1'/'payload.zip'
        result=holders[0].call('verified_open',**common,issued=issued,registry_public=admin.info['public_key'],
                               entry=encode_entry(entries[0]),output_path=str(outfile))
        assert outfile.read_bytes()==payloads[0] and result['payload_path']==str(outfile.resolve())
        assert os.stat(outfile).st_mode & 0o777==0o600
        with pytest.raises(RoleError):
            holders[0].call('verified_open',**common,issued=issued,registry_public=admin.info['public_key'],
                            entry=encode_entry(entries[0]),output_path=str(outfile))

        payload_paths=[]
        for i,payload in enumerate(payloads):
            path=tmp_path/f'input-{i}.zip';path.write_bytes(payload);payload_paths.append(str(path))
        outer_path=tmp_path/'issuer'/'outer.bin'
        file_pack=issuer.call('make_and_seal_files',enrollment=enrollment,admission=adm,recipients=recipients,
                              session_secret=coeff[0],payload_paths=payload_paths,outer_path=str(outer_path))
        assert outer_path.is_file() and 'outer' not in file_pack
        file_entries,file_manifest=open_bundle(gate,admission,recipients,outer_path.read_bytes(),
                                                recovered,issuer.public_key())
        file_common={**common,'manifest':file_manifest,'issuer_signature':file_pack['signature']}
        file_issued=admin.call('create_issuance',**file_common)
        cipher_path=tmp_path/'delivery'/'0.cipher';cipher_path.parent.mkdir()
        wrapped_path=tmp_path/'delivery'/'0.key'
        cipher_path.write_bytes(file_entries[0]['ciphertext']);wrapped_path.write_bytes(file_entries[0]['wrapped_key'])
        file_entry={'context':file_entries[0]['context'],'cipher_path':str(cipher_path),
                    'wrapped_path':str(wrapped_path)}
        result=holders[0].call('verified_open',**file_common,issued=file_issued,
                               registry_public=admin.info['public_key'],entry=file_entry)
        assert result['recipient']==1 and result['payload_bytes']==len(payloads[0])
        with pytest.raises(RoleError):
            issuer.call('make_and_seal_files',enrollment=enrollment,admission=adm,recipients=recipients,
                        session_secret=coeff[0],payload_paths=payload_paths,outer_path=str(outer_path))


def test_owner_only_file_write_will_not_overwrite_existing_output(tmp_path):
    # File creation semantics are exercised in the full test with generated roles;
    # this test checks the reusable owner-only file primitive and overwrite guard.
    from tvmc.process_roles import _write_new
    path=tmp_path/'fresh'/'package.bin'
    assert _write_new(path,b'first')==str(path.resolve())
    with pytest.raises(FileExistsError):
        _write_new(path,b'changed')
    assert path.read_bytes()==b'first' and os.stat(path).st_mode & 0o777==0o600


def test_issuer_builds_merged_fixture_models_in_its_own_process(tmp_path):
    # One optimization step tests the IPC/model packaging path only; it is not
    # an assertion of successful fingerprint attribution or model quality.
    words=[[0]*256,[1]*256,[i%2 for i in range(256)]]
    with RoleProcess('issuer') as issuer:
        result=issuer.call('build_payloads',output_dir=str(tmp_path/'models'),words=words,
                           replica_ids=['fixture-U1','fixture-U2','fixture-U3'],
                           offline_fixture=True,steps=1,threads=1)
        assert result['scope']=='synthetic_fixture' and result['training_count']==3
        assert len(result['prompts'])==768
        for replica in result['replicas']:
            with zipfile.ZipFile(replica['path']) as archive:
                assert 'base/model.safetensors' in archive.namelist()
                assert 'adapter.pt' not in archive.namelist()
