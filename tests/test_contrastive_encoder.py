from types import SimpleNamespace
import torch
from transformers import BertConfig, BertModel
from experiments.contrastive_encoder_benchmark import tune


def test_last_block_training_preserves_base_and_restores_outputs(tmp_path):
    torch.manual_seed(0)
    model = BertModel(BertConfig(vocab_size=32, hidden_size=12, num_hidden_layers=2,
        num_attention_heads=3, intermediate_size=24, hidden_dropout_prob=0, attention_probs_dropout_prob=0))
    original = {k: v.clone() for k, v in model.state_dict().items()}
    inputs = {'input_ids': torch.tensor([[1, 2, 0], [1, 3, 0], [4, 5, 6]]),
              'attention_mask': torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 1]])}
    encoder = SimpleNamespace(model=model, tokenizer=lambda *args, **kwargs: inputs)
    examples = [{'intent': 'a', 'rules': [{'kind': 'unchanged', 'parameter': 'source'}]},
                {'intent': 'b', 'rules': [{'kind': 'unchanged', 'parameter': 'source'}]},
                {'intent': 'c', 'rules': None}]
    history = tune(encoder, examples, steps=2)
    assert torch.isfinite(torch.tensor(history[0]['loss']))
    after = model.state_dict()
    assert all(torch.equal(v, after[k]) for k, v in original.items() if not k.startswith('encoder.layer.1.'))
    assert any(not torch.equal(v, after[k]) for k, v in original.items() if k.startswith('encoder.layer.1.'))
    assert all(not p.requires_grad for p in model.parameters())
    checkpoint = tmp_path / 'block.pt'
    torch.save(model.encoder.layer[-1].state_dict(), checkpoint)
    fresh = BertModel(model.config).eval()
    fresh.load_state_dict(original)
    fresh.encoder.layer[-1].load_state_dict(torch.load(checkpoint, weights_only=True))
    with torch.no_grad():
        assert torch.allclose(model(**inputs).last_hidden_state, fresh(**inputs).last_hidden_state)
