import numpy as np
import pytest
from dbmf.models import FactorModel
from dbmf.artifacts import save
from dbmf.data import Dataset

def test_authenticated_endpoint(tmp_path, monkeypatch):
    pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    from dbmf.api import create_app
    vocabulary = {'gender': ['F'], 'age_band': ['0-35'], 'highest_education': ['A Level']}
    data = Dataset([], ['A:2014B', 'B:2014B'], [], vocabulary, np.empty((0,3)),
                   np.empty((0,2)), np.empty((0,2)), {}, {}, 1, {})
    save(FactorModel('dbmf', 1, 2, 3), data, tmp_path/'model', {})
    monkeypatch.setenv('DBMF_MODEL_DIR', str(tmp_path/'model'))
    monkeypatch.setenv('DBMF_API_TOKEN', 'x'*32)
    with TestClient(create_app()) as client:
        body = {'profile': {'gender': 'F', 'age_band': '0-35', 'highest_education': 'A Level'}}
        assert client.get('/health').status_code == 200
        assert client.post('/recommend', json=body).status_code == 401
        header = {'Authorization': 'Bearer '+'x'*32}
        assert client.post('/recommend', json=body, headers=header).status_code == 200
        body['k'] = 0
        assert client.post('/recommend', json=body, headers=header).status_code == 422

def test_fails_without_secret(monkeypatch):
    pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    from dbmf.api import create_app
    monkeypatch.delenv('DBMF_API_TOKEN', raising=False)
    with pytest.raises(RuntimeError, match='TOKEN'):
        with TestClient(create_app()):
            pass
