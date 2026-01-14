# RecBole Experiments: GRU4Rec and SASRec on Multiple Datasets

This repo runs GRU4Rec and SASRec on MovieLens (1M/20M), Amazon (Books, Sports, Beauty, Electronics) and Yelp using RecBole.

## Setup
1. Install deps: `pip install -r requirements.txt`
2. Download datasets to `data/` (see links in code comments).
3. Run: `python run_all.py`

## Configs
- `general_config.yaml`: Shared settings.
- Model-specific: Overrides for each.

Results in `results/`. Logs in `log/`.