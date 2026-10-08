import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import load, encode, FIELDS
from .training import train
from .evaluation import ranking, metrics, summarize, paired_bootstrap
from .artifacts import save

def evaluate(data, score):
    return np.asarray([metrics(ranking(score(u)), data.cold[u]) for u in sorted(data.cold)])

def run(directory, output, seeds=(42,), epochs=100, ablations=False, registration_mode='relative', starts=None, training_config=None, missing_registration='error'):
    options = dict(rank=20, batch_size=512, learning_rate=.001, weight_decay=.0001, patience=10)
    if training_config:
        if set(training_config) - set(options):
            raise ValueError('Unsupported training configuration keys')
        options.update(training_config)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    report = {'protocol': {'registration_mode': registration_mode, 'missing_registration': missing_registration, 'seeds': list(seeds),
                          **options, 'k': 5, 'epochs_limit': epochs,
                          'negative_samples_per_positive': 1, 'cpu_threads': 1},
              'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                           'numpy': np.__version__, 'pandas': pd.__version__},
              'sources': {name: hashlib.sha256((Path(directory)/name).read_bytes()).hexdigest()
                          for name in ('studentInfo.csv', 'studentRegistration.csv')}, 'runs': []}
    for seed in seeds:
        data = load(directory, seed, registration_mode, starts, missing_registration)
        result = {'seed': seed, 'counts': data.audit, 'models': {}, 'subgroups': {}, 'ablations': {}}
        pop = np.bincount(data.train[:, 1], minlength=len(data.items))
        groups = {tuple(profile[f] for f in FIELDS): np.zeros(len(data.items))
                  for profile in data.demographics[:data.warm_count]}
        for u, i in data.train:
            key = tuple(data.demographics[u][f] for f in FIELDS)
            groups.setdefault(key, np.zeros(len(data.items)))[i] += 1
        scores = {'pop': evaluate(data, lambda u: pop),
                  'demog_pop': evaluate(data, lambda u: groups.get(tuple(data.demographics[u][f] for f in FIELDS), pop))}
        for kind in ('bmf', 'dbmf', 'concat', 'fm'):
            model, history = train(data, kind=kind, seed=seed, epochs=epochs, **options)
            scores[kind] = evaluate(data, lambda u: model.score(-1, data.features[u]))
            result['models'][kind] = {'history': history}
            warm_relevant = {}
            known = {}
            for u, i in data.validation:
                warm_relevant.setdefault(int(u), set()).add(int(i))
            for u, i in data.train:
                known.setdefault(int(u), set()).add(int(i))
            result['models'][kind]['warm_validation'] = summarize([
                metrics(ranking(model.score(u, data.features[u]), excluded=known.get(u, ())), rel)
                for u, rel in sorted(warm_relevant.items())])
            if kind == 'dbmf':
                save(model, data, output/f'model-{seed}', {'seed': seed, 'counts': data.audit, 'protocol': report['protocol']})
        for kind, rows in scores.items():
            result['models'].setdefault(kind, {})['cold'] = summarize(rows)
        result['paired_bootstrap_dbmf_vs_bmf'] = paired_bootstrap(scores['dbmf'], scores['bmf'], seed)
        cold_users = sorted(data.cold)
        for field in FIELDS:
            result['subgroups'][field] = {}
            for value in sorted({data.demographics[u][field] for u in cold_users}):
                indices = [i for i, u in enumerate(cold_users) if data.demographics[u][field] == value]
                result['subgroups'][field][value] = {'n': len(indices), 'metrics': summarize(scores['dbmf'][indices])}
        if ablations:
            from itertools import combinations
            for size in range(1, 4):
                for active in combinations(FIELDS, size):
                    features = encode(data.demographics, data.vocabulary, active)
                    model, _ = train(data, seed=seed, epochs=epochs, features=features, **options)
                    result['ablations']['+'.join(active)] = summarize(evaluate(data, lambda u: model.score(-1, features[u])))
        report['runs'].append(result)
        (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
