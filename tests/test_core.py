import json
import numpy as np
import pandas as pd
import pytest
import torch
from dbmf.models import FactorModel
from dbmf.data import load, encode, FIELDS
from dbmf.evaluation import metrics, ranking
from dbmf.training import train
from dbmf.artifacts import save, Recommender
from dbmf.experiment import run

@pytest.fixture
def dataset(tmp_path):
    rows = []
    for u in range(30):
        for i in range(3):
            rows.append(dict(id_student=u, code_module=chr(65+i), code_presentation='2014B',
                             gender='M' if u%2 else 'F', age_band='0-35', highest_education='A Level',
                             final_result='Pass' if i == u%3 else 'Fail', date_registration=u))
    frame = pd.DataFrame(rows)
    frame.drop(columns='date_registration').to_csv(tmp_path/'studentInfo.csv', index=False)
    frame[['id_student','code_module','code_presentation','date_registration']].to_csv(tmp_path/'studentRegistration.csv', index=False)
    return tmp_path

def test_metrics_and_stable_ties():
    assert ranking([1,1,0], 2).tolist() == [0,1]
    assert metrics([2,1,0], {2}, 3).tolist() == [1/3, 1, 1]
    assert ranking([1,2,3], 5, {0,1,2}).size == 0

@pytest.mark.parametrize('kind', ['bmf','dbmf','concat','fm'])
def test_equations(kind):
    model = FactorModel(kind, 2, 3, 3, 2)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1)
    d = torch.tensor([[1.,0.,1.]])
    cold = model(torch.tensor([-1]), torch.tensor([0]), d).item()
    warm = model(torch.tensor([0]), torch.tensor([0]), d).item()
    expected = {'bmf': (2,5), 'dbmf': (6,9), 'concat': (4,7), 'fm': (10,17)}
    assert (cold,warm) == expected[kind]
    batch = model.score_batch([-1,0], np.asarray([[1.,0.,1.],[1.,0.,1.]], dtype=np.float32))
    assert np.allclose(batch[0], cold)
    assert np.allclose(batch[1], warm)

def test_split_encoder_and_determinism(dataset):
    data = load(dataset)
    assert data.warm_count == 24
    assert max(data.train[:,0]) < min(data.cold)
    assert not set(map(tuple, data.train)) & set(map(tuple, data.validation))
    assert np.array_equal(load(dataset).train, data.train)
    assert encode([{}], data.vocabulary).sum() == 0
    first, _ = train(data, epochs=2)
    second, _ = train(data, epochs=2)
    assert np.array_equal(first.score(-1, data.features[-1]), second.score(-1, data.features[-1]))

def test_artifact_and_tamper(dataset, tmp_path):
    data = load(dataset); model, _ = train(data, epochs=2)
    destination = tmp_path/'model'
    save(model, data, destination, {})
    service = Recommender(destination)
    assert np.allclose(model.score(-1, data.features[-1]), service.model.score(-1, data.features[-1]))
    result = service.recommend(data.demographics[-1], eligible=[data.items[0]])
    assert len(result['recommendations']) == 1
    with np.load(destination/'weights.npz') as arrays:
        assert not any(k.startswith('user') for k in arrays.files)
    with (destination/'weights.npz').open('ab') as stream:
        stream.write(b'bad')
    with pytest.raises(ValueError, match='checksum'):
        Recommender(destination)

def test_full_experiment(dataset, tmp_path):
    report = run(dataset, tmp_path/'output', epochs=2, ablations=True)
    assert len(report['runs'][0]['models']) == 6
    assert len(report['runs'][0]['ablations']) == 7
    assert json.loads((tmp_path/'output/report.json').read_text())['runs']
