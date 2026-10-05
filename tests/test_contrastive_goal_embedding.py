import pytest
import torch
import torch.nn.functional as F
from experiments.contrastive_goal_embedding import fit, ProjectedEncoder


def test_pair_training_changes_geometry_and_reloads_without_fitting(tmp_path):
    vectors = torch.tensor([[1., 0.1, 0], [1., -0.1, 0], [1., 0, 0.2], [1., 0, -0.2]])
    examples = [{'rules': [{'kind': 'unchanged', 'parameter': 'source'}]}] * 2 + [{'rules': [{'kind': 'absent', 'parameter': 'source'}]}] * 2
    before = vectors.clone()
    weight, history = fit(vectors, examples, steps=100)
    assert torch.equal(vectors, before)
    assert history[-1]['loss'] < history[0]['loss']
    path = tmp_path / 'projection.pt'
    torch.save({'weight': weight}, path)
    restored = torch.load(path, weights_only=True)['weight']
    encoder = ProjectedEncoder(lambda texts: vectors[:len(texts)], restored)
    assert torch.allclose(encoder(['a', 'b']), F.normalize(vectors[:2] @ weight.T, dim=1))
    assert torch.allclose(encoder(['a', 'b']).norm(dim=1), torch.ones(2))


def test_pair_training_requires_both_pair_types():
    with pytest.raises(ValueError, match='positive and negative'):
        fit(torch.eye(2), [{'rules': None}, {'rules': None}], steps=1)


@pytest.mark.parametrize('weight', [torch.ones(2, 3), torch.tensor([[float('nan')]])])
def test_invalid_projection_rejected(weight):
    with pytest.raises(ValueError):
        ProjectedEncoder(lambda texts: torch.eye(2), weight)
