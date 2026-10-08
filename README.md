# OULAD DB-MF

DB-MF, B-MF, Concat-MF, FM, POP and Demog-POP for cold-start course recommendation, with training, evaluation and an authenticated inference API.

```sh
python -m pip install -e ".[test,api]"
python tools/fetch_oulad.py
dbmf experiment --data data/oulad --output results --seeds 42 --ablations --missing-registration exclude
dbmf recommend --model results/model-42 --profile examples/profile.json
python -m pytest -q
```

The default split orders presentation-relative registration offsets, as interpreted from the manuscript. For absolute chronology, use `--registration-mode absolute --presentation-starts starts.json` with actual start dates. Unspecified paper settings use configurable defaults; identical published scores are not guaranteed.

For the API, set `DBMF_MODEL_DIR` and `DBMF_API_TOKEN` (at least 32 characters), then run `uvicorn dbmf.api:create_app --factory`. Send a bearer token to `POST /recommend` with `profile`, optional `k`, `eligible`, and `excluded`. Scores are ranking values, not success probabilities.

MIT license. Downloaded OULAD data and the original manuscript retain their separate terms.
