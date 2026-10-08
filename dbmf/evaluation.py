import numpy as np

def ranking(scores, k=5, excluded=()):
    scores = np.asarray(scores, dtype=float).copy()
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError('Scores must be a finite vector')
    if k < 1:
        raise ValueError('k must be positive')
    candidates = np.asarray([i for i in range(len(scores)) if i not in excluded], dtype=np.int64)
    return candidates[np.lexsort((candidates, -scores[candidates]))][:k]

def metrics(recommended, relevant, k=5):
    if not relevant:
        raise ValueError('Metrics require a relevant item')
    hits = np.asarray([i in relevant for i in recommended], dtype=float)
    discounts = 1 / np.log2(np.arange(len(hits)) + 2)
    ideal = (1 / np.log2(np.arange(min(k, len(relevant))) + 2)).sum()
    return np.asarray([hits.sum()/k, hits.sum()/len(relevant), (hits*discounts).sum()/ideal])

def summarize(rows):
    values = np.asarray(rows)
    return {name: {'mean': float(values[:, j].mean()), 'std': float(values[:, j].std())}
            for j, name in enumerate(('precision', 'recall', 'ndcg'))}

def paired_bootstrap(a, b, seed=42, replicates=2000):
    delta = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, delta.shape[1]))
    for i in range(replicates):
        draws[i] = delta[rng.integers(len(delta), size=len(delta))].mean(0)
    return {'mean_difference': delta.mean(0).tolist(),
            'ci95': np.quantile(draws, [.025, .975], axis=0).T.tolist(),
            'replicates': replicates, 'seed': seed}
