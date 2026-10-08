import torch
from torch import nn

class FactorModel(nn.Module):
    """Equations 1-4, Concat-MF and dense second-order FM; -1 denotes cold user."""
    def __init__(self, kind, users, items, width, rank=20):
        super().__init__()
        if kind not in ('bmf', 'dbmf', 'concat', 'fm'):
            raise ValueError('Unknown model')
        self.kind, self.users, self.items, self.width, self.rank = kind, users, items, width, rank
        self.global_bias = nn.Parameter(torch.zeros(()))
        self.user = nn.Embedding(users, rank)
        self.item = nn.Embedding(items, rank + (width if kind == 'concat' else 0))
        self.user_bias = nn.Embedding(users, 1)
        self.item_bias = nn.Embedding(items, 1)
        self.projection = nn.Linear(width, rank, bias=False) if kind in ('dbmf', 'fm') else None
        self.demographic_bias = nn.Parameter(torch.zeros(width)) if kind == 'fm' else None
        for module in (self.user, self.item):
            nn.init.normal_(module.weight, std=.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        if self.projection is not None:
            nn.init.normal_(self.projection.weight, std=.01)

    def forward(self, users, items, demographics):
        warm = (users >= 0).to(demographics.dtype)
        p = self.user(users.clamp_min(0)) * warm[:, None]
        q = self.item(items)
        bias = self.global_bias + self.item_bias(items).squeeze(-1) + self.user_bias(users.clamp_min(0)).squeeze(-1) * warm
        if self.kind == 'dbmf':
            p = p + self.projection(demographics)
        elif self.kind == 'concat':
            p = torch.cat((p, demographics), dim=-1)
        elif self.kind == 'fm':
            # FM trick includes each active demographic separately, and user/item.
            v = self.projection.weight.T
            dsum = demographics @ v
            squared = p.square() + q.square() + demographics.square() @ v.square()
            interactions = .5 * ((p + q + dsum).square() - squared).sum(-1)
            return bias + demographics @ self.demographic_bias + interactions
        return bias + (p * q).sum(-1)

    @torch.inference_mode()
    def score_batch(self, users, demographics):
        users = torch.as_tensor(users, dtype=torch.long)
        demographics = torch.as_tensor(demographics, dtype=torch.float32)
        count = len(users)
        u = users[:, None].expand(count, self.items).reshape(-1)
        i = torch.arange(self.items).expand(count, self.items).reshape(-1)
        d = demographics[:, None, :].expand(count, self.items, self.width).reshape(-1, self.width)
        return self(u, i, d).reshape(count, self.items).numpy()

    @torch.inference_mode()
    def score(self, user, demographics):
        d = torch.as_tensor(demographics, dtype=torch.float32).expand(self.items, -1)
        return self(torch.full((self.items,), user, dtype=torch.long), torch.arange(self.items), d).numpy()
