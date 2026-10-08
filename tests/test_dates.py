import numpy as np
import pandas as pd
import pytest
from dbmf.data import load

def test_missing_date_policy(tmp_path):
    rows = [dict(id_student=u, code_module='A', code_presentation='2014B',
                 gender='F', age_band='0-35', highest_education='A Level',
                 final_result='Pass', date_registration=np.nan if u == 9 else u) for u in range(10)]
    frame = pd.DataFrame(rows)
    frame.drop(columns='date_registration').to_csv(tmp_path/'studentInfo.csv', index=False)
    frame[['id_student','code_module','code_presentation','date_registration']].to_csv(tmp_path/'studentRegistration.csv', index=False)
    with pytest.raises(ValueError, match='registration'):
        load(tmp_path)
    data = load(tmp_path, missing_registration='exclude')
    assert data.audit['excluded_undated_positive_users'] == 1
    assert len(data.users) == 9
    absolute = load(tmp_path, registration_mode='absolute', starts={'2014B': '2014-02-01'}, missing_registration='exclude')
    assert absolute.users == data.users
    with pytest.raises(ValueError, match='mapping'):
        load(tmp_path, registration_mode='absolute')
