import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from .data import encode
from .models import FactorModel
from .evaluation import ranking

def save(model, data, directory, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Inference artifact deliberately contains no student identifiers or user factors.
    config = {'schema_version': 1, 'kind': model.kind, 'items': data.items,
              'vocabulary': data.vocabulary, 'rank': model.rank, 'metadata': metadata}
    arrays = {k: v.detach().numpy() for k, v in model.state_dict().items()
              if not k.startswith(('user.', 'user_bias.'))}
    temp = directory / 'weights.tmp.npz'
    np.savez_compressed(temp, **arrays)
    os.replace(temp, directory / 'weights.npz')
    config['weights_sha256'] = hashlib.sha256((directory / 'weights.npz').read_bytes()).hexdigest()
    temporary = directory / 'manifest.tmp.json'
    temporary.write_text(json.dumps(config, indent=2), encoding='utf-8')
    os.replace(temporary, directory / 'manifest.json')

class Recommender:
    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest['schema_version'] != 1:
            raise ValueError('Unsupported artifact schema')
        path = directory / 'weights.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.manifest['weights_sha256']:
            raise ValueError('Artifact checksum mismatch')
        self.items = self.manifest['items']
        width = sum(map(len, self.manifest['vocabulary'].values()))
        self.model = FactorModel(self.manifest['kind'], 1, len(self.items), width, self.manifest['rank'])
        with np.load(path, allow_pickle=False) as arrays:
            state = self.model.state_dict()
            for key in state:
                if not key.startswith(('user.', 'user_bias.')):
                    state[key] = torch.from_numpy(arrays[key].copy())
            self.model.load_state_dict(state)
        self.model.eval()

    def recommend(self, profile, k=5, eligible=None, excluded=()):
        unknown = [f for f, categories in self.manifest['vocabulary'].items()
                   if str(profile.get(f, '')) not in categories]
        d = encode([profile], self.manifest['vocabulary'])[0]
        scores = self.model.score(-1, d)
        allowed = set(self.items if eligible is None else eligible)
        if allowed - set(self.items) or set(excluded) - set(self.items):
            raise ValueError('Unknown catalog item')
        blocked = {i for i, name in enumerate(self.items) if name not in allowed or name in excluded}
        indices = ranking(scores, k, blocked)
        return {'recommendations': [{'item': self.items[i], 'score': float(scores[i])} for i in indices],
                'unknown_demographic_fields': unknown,
                'score_semantics': 'ranking score, not a probability of academic success'}
