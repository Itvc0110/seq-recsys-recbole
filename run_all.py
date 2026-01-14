from recbole.quick_start import run_recbole
from recbole.config import Config
import os

models = ['GRU4Rec', 'SASRec']
datasets = ['ml-1m', 'ml-20m', 'Books', 'Sports_and_Outdoors', 'Beauty', 'Electronics', 'Yelp']
config_files = {
    'GRU4Rec': ['configs/general_config.yaml', 'configs/gru4rec.yaml'],
    'SASRec': ['configs/general_config.yaml', 'configs/sasrec.yaml']
}

os.makedirs('results', exist_ok=True)

for model in models:
    for dataset in datasets:
        print(f"Running {model} on {dataset}...")
        config = Config(model=model, dataset=dataset, config_file_list=config_files[model])
        
        if dataset in ['ml-20m', 'Electronics']:  # Larger datasets
            config['MAX_ITEM_LIST_LENGTH'] = 200
            config['train_batch_size'] = 512
        
        result = run_recbole(config_dict=config.final_config_dict)
        with open(f'results/{model}_{dataset}.txt', 'w') as f:
            f.write(str(result))
        print(f"Completed {model} on {dataset}. Results: {result}")