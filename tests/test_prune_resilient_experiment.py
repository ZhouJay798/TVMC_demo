import numpy as np
import torch

from run_prune_resilient_experiment import secret_pairs, train_rows
from tvmc.robustness import aggregate, aggregate_with_abstention


def test_private_pair_mapping_is_reproducible_and_excludes_primary_tokens():
    a, slots = secret_pairs(b'a'*32, 100, {0,1,2})
    assert (a, slots) == secret_pairs(b'a'*32, 100, {0,1,2})
    assert len(a) == len(set(a)) == 32 and not set(a).intersection({0,1,2})
    assert len(slots) == 256 and all(0 <= s < 16 for s in slots)
    assert (a, slots) != secret_pairs(b'b'*32, 100, {0,1,2})


def test_alternate_row_training_improves_its_training_loss():
    torch.manual_seed(19)
    # Every bit can be classified by a simple sign feature.
    labels = np.array([i % 2 for i in range(256)])
    sign = torch.tensor(labels*2-1,dtype=torch.float32)
    features = torch.zeros((256,3,4))
    features[:,:,0] = sign[:,None]
    initial = torch.zeros((32,4))
    utility = torch.zeros((8,4))
    rows, info = train_rows(initial,[features],utility,labels,[i % 16 for i in range(256)],steps=20,lr=.02)
    assert info['loss_first_last'][1] < info['loss_first_last'][0]
    assert torch.linalg.vector_norm(rows) > 0


def test_zero_and_weak_margins_are_erasures_without_changing_legacy():
    values=[0.,0.,0., .2,.2,.2, -.3,-.3,-.3, np.nan,np.nan,np.nan]
    assert aggregate(values,3).tolist()==[1,1,0,-1]
    assert aggregate_with_abstention(values,3,min_abs_margin=.25).tolist()==[-1,-1,0,-1]
