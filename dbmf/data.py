from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

FIELDS = ('gender', 'age_band', 'highest_education')
KEYS = ['id_student', 'code_module', 'code_presentation']

@dataclass
class Dataset:
    users: list
    items: list
    demographics: list
    vocabulary: dict
    features: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    cold: dict
    positives: dict
    warm_count: int
    audit: dict

def encode(profiles, vocabulary, active=FIELDS):
    offsets = {}; size = 0
    for field in FIELDS:
        offsets[field] = size
        size += len(vocabulary[field])
    result = np.zeros((len(profiles), size), dtype=np.float32)
    for row, profile in enumerate(profiles):
        for field in active:
            value = str(profile.get(field, ''))
            if value in vocabulary[field]:
                result[row, offsets[field] + vocabulary[field].index(value)] = 1
    return result

def load(directory, seed=42, registration_mode='relative', starts=None, missing_registration='error'):
    """Split positive-user population, using ALL enrollments for first registration."""
    directory = Path(directory)
    info = pd.read_csv(directory / 'studentInfo.csv')
    reg = pd.read_csv(directory / 'studentRegistration.csv')
    for frame, required in [(info, KEYS + list(FIELDS) + ['final_result']),
                            (reg, KEYS + ['date_registration'])]:
        missing = set(required) - set(frame)
        if missing:
            raise ValueError(f'Missing columns: {sorted(missing)}')
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError('Enrollment keys must be complete and unique')
    merged = info.merge(reg[KEYS + ['date_registration']], on=KEYS, how='left', validate='one_to_one')
    merged['item'] = merged.code_module.astype(str) + ':' + merged.code_presentation.astype(str)
    dates = pd.to_numeric(merged.date_registration, errors='coerce')
    missing_date_rows = int(dates.isna().sum())
    if registration_mode == 'absolute':
        if not starts:
            raise ValueError('Absolute registration mode requires presentation-start mapping')
        base = merged.apply(lambda r: starts.get(r['item'], starts.get(str(r.code_presentation))), axis=1)
        if base.isna().any():
            raise ValueError('Presentation-start mapping is incomplete')
        dates = pd.to_datetime(base, errors='raise') + pd.to_timedelta(dates, unit='D')
    elif registration_mode != 'relative':
        raise ValueError('Unknown registration mode')
    merged['order_date'] = dates
    positive = merged[merged.final_result.isin(['Pass', 'Distinction'])]
    population = set(positive.id_student)
    earliest = merged[merged.id_student.isin(population)].groupby('id_student').order_date.min()
    excluded_users = int(earliest.isna().sum())
    if excluded_users:
        if missing_registration == 'exclude':
            earliest = earliest.dropna()
            positive = positive[positive.id_student.isin(earliest.index)]
        else:
            raise ValueError('A positive student has no registration date; supply corrected source data or explicitly select missing-registration exclusion')
    users = earliest.reset_index().sort_values(['order_date', 'id_student']).id_student.tolist()
    warm_count = int(len(users) * .8)
    if warm_count < 2 or warm_count == len(users):
        raise ValueError('Insufficient users for temporal split')
    user_map = {u: i for i, u in enumerate(users)}
    items = sorted(merged.item.unique().tolist())
    item_map = {v: i for i, v in enumerate(items)}
    # First available enrollment profile, never a later outcome-dependent profile.
    profiles = merged.sort_values(['order_date', 'id_student', 'item']).drop_duplicates('id_student').set_index('id_student')
    demographics = [{f: '' if pd.isna(profiles.loc[u, f]) else str(profiles.loc[u, f]) for f in FIELDS} for u in users]
    vocabulary = {f: sorted({p[f] for p in demographics[:warm_count] if p[f]}) for f in FIELDS}
    edges = sorted({(user_map[r.id_student], item_map[r.item]) for r in positive.itertuples()})
    positives = {u: set() for u in range(len(users))}
    for u, i in edges:
        positives[u].add(i)
    warm = np.asarray([(u, i) for u, i in edges if u < warm_count], dtype=np.int64)
    rng = np.random.default_rng(seed)
    warm = warm[rng.permutation(len(warm))]
    nval = max(1, int(np.ceil(len(warm) * .1)))
    if nval >= len(warm):
        raise ValueError('Insufficient warm edges')
    validation, train = warm[:nval], warm[nval:]
    cold = {u: positives[u] for u in range(warm_count, len(users))}
    audit = {'users': len(users), 'items': len(items), 'positives': len(edges),
             'warm_users': warm_count, 'cold_users': len(cold), 'train_edges': len(train),
             'validation_edges': len(validation), 'registration_mode': registration_mode,
             'missing_registration_rows': missing_date_rows,
             'excluded_undated_positive_users': excluded_users,
             'demographic_width': sum(map(len, vocabulary.values()))}
    return Dataset(users, items, demographics, vocabulary, encode(demographics, vocabulary),
                   train, validation, cold, positives, warm_count, audit)
