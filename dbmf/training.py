from copy import deepcopy
import numpy as np
import torch
from .evaluation import ranking, metrics
from .models import FactorModel

def train(data, kind='dbmf', seed=42, rank=20, epochs=100, batch_size=512,
          learning_rate=.001, weight_decay=.0001, patience=10, features=None):
    if epochs < 1 or batch_size < 1 or patience < 1 or learning_rate <= 0 or weight_decay < 0 or rank < 1:
        raise ValueError('Invalid training configuration')
    torch.manual_seed(seed)
    # Small catalog and embedding tables: one CPU thread avoids dispatch overhead.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    features = data.features if features is None else features
    model = FactorModel(kind, data.warm_count, len(data.items), features.shape[1], rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Exclude all known positives including warm validation, never label held-out successes negative.
    pools = {u: np.asarray(sorted(set(range(len(data.items))) - data.positives[u]), dtype=np.int64)
             for u in range(data.warm_count)}
    edges = np.asarray([(u, i) for u, i in data.train if len(pools[u])], dtype=np.int64)
    if not len(edges):
        raise ValueError('No trainable edges have eligible negative items')
    known = {}
    relevant = {}
    for u, i in data.train:
        known.setdefault(int(u), set()).add(int(i))
    for u, i in data.validation:
        relevant.setdefault(int(u), set()).add(int(i))
    best, state, stale, history = -float('inf'), None, 0, []
    for epoch in range(epochs):
        model.train(); losses = []
        shuffled = edges[rng.permutation(len(edges))]
        for start in range(0, len(edges), batch_size):
            batch = shuffled[start:start+batch_size]
            u = torch.tensor(batch[:, 0]); pos = torch.tensor(batch[:, 1])
            neg = torch.tensor([rng.choice(pools[int(v)]) for v in batch[:, 0]])
            d = torch.tensor(features[batch[:, 0]])
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.softplus(-(model(u, pos, d) - model(u, neg, d))).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        validation_users = sorted(relevant)
        all_scores = model.score_batch(validation_users, features[validation_users])
        ndcg = np.mean([metrics(ranking(scores, excluded=known.get(u, ())), relevant[u])[2]
                        for u, scores in zip(validation_users, all_scores)])
        history.append({'epoch': epoch+1, 'loss': float(np.mean(losses)), 'validation_ndcg5': float(ndcg)})
        if ndcg > best:
            best, state, stale = ndcg, deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(state); model.eval()
    return model, history
