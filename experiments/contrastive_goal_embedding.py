"""Learn an embedding metric from goal pairs; no generative LLM or text rules."""
import torch
import torch.nn.functional as F
from vectorpro.acquisition import digest


def fit(vectors, examples, steps=400):
    x = F.normalize(vectors.detach().cpu().float().clone(), dim=1)
    if x.ndim != 2 or len(x) != len(examples) or not torch.isfinite(x).all():
        raise ValueError('Invalid training matrix')
    labels = [digest(e['rules']) for e in examples]
    positive = torch.tensor([[a == b for b in labels] for a in labels])
    diagonal = torch.eye(len(x), dtype=torch.bool)
    positive &= ~diagonal
    negative = ~positive & ~diagonal
    if not positive.any() or not negative.any():
        raise ValueError('Contrast learning requires positive and negative pairs')
    identity = torch.eye(x.shape[1])
    weight = torch.nn.Parameter(identity.clone())
    optimizer = torch.optim.Adam([weight], lr=0.01)
    history = []
    for step in range(steps):
        z = F.normalize(x @ weight.T, dim=1)
        similarities = z @ z.T
        # Same goal attracts; different goals repel. Balanced pair means avoid class-size dominance.
        loss = (1 - similarities[positive]).square().mean() + similarities[negative].square().mean()
        loss = loss + 0.001 * (weight - identity).square().mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0:
            history.append({'step': step + 1, 'loss': float(loss.detach())})
    return weight.detach(), history


class ProjectedEncoder:
    def __init__(self, encoder, weight):
        if weight.ndim != 2 or weight.shape[0] != weight.shape[1] or not torch.isfinite(weight).all():
            raise ValueError('Invalid metric projection')
        self.encoder, self.weight = encoder, weight.detach().cpu().float().clone()

    def __call__(self, texts):
        vectors = self.encoder(texts).detach().cpu().float()
        if vectors.ndim != 2 or vectors.shape[1] != self.weight.shape[1] or not torch.isfinite(vectors).all():
            raise ValueError('Encoder/projector mismatch')
        return F.normalize(vectors @ self.weight.T, dim=1)
