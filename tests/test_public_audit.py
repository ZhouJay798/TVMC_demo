import json
import numpy as np
from tvmc.experiment import core_experiment
from tvmc.public_audit import audit_releases
from verify_evidence import verify


def test_all_registered_release_evidence_and_wrong_user_rejections(tmp_path):
    _, state = core_experiment(trials=2)
    class FixtureLab:
        def restore(self, i): self.i=i
        def predict(self, prompts): return state['words'][self.i][:len(prompts)]
    rows=audit_releases(FixtureLab(),list(range(5)),list(range(256)),state,'fixture-asset',tmp_path)
    assert len(rows)==5 and all(r['passed'] for r in rows)
    for i in range(1,6):
        folder=tmp_path/'public_audit'/f'user_{i}'
        context=json.loads((folder/'public_context.json').read_text())
        evidence=json.loads((folder/'public_evidence.json').read_text())
        assert all(verify(context,evidence).values())
        # Public-only replay requires no private registry/key state.
        context['key_context']['sha256']='different-asset'
        assert not verify(context,evidence)['key_layer']
