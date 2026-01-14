from recbole.quick_start import run_recbole
from recbole.config import Config
import os
import traceback
import torch

models = ['GRU4Rec', 'SASRec']
datasets = ['ml-1m', 'ml-20m', 'Books', 'Sports_and_Outdoors', 'Beauty', 'Electronics', 'yelp']

config_files = {
    'GRU4Rec': ['configs/general_config.yaml', 'configs/gru4rec_config.yaml'],
    'SASRec': ['configs/general_config.yaml', 'configs/sasrec_config.yaml']
}

os.makedirs('results', exist_ok=True)

for model in models:
    for dataset in datasets:
        dataset_dir = f"./data/{dataset}"
        inter_file = f"{dataset_dir}/{dataset}.inter"
        
        if not (os.path.exists(dataset_dir) and os.path.exists(inter_file)):
            print(f"Skipping {model} on {dataset}: Dataset folder or .inter file not found.")
            continue
        
        print(f"Running {model} on {dataset}...")
        
        try:
            config = Config(model=model, dataset=dataset, config_file_list=config_files[model])
            
            # Force single process + disable any distributed
            config['nproc'] = 1
            config['world_size'] = 1
            config['local_rank'] = 0
            config['use_gpu'] = True if torch.cuda.is_available() else False
            
            # IMPORTANT: Disable checkpoint loading for fresh training
            config['load_collected'] = False
            config['resume_training'] = False
            
            # Prevent any model_file loading bug
            if 'model_file' in config.final_config_dict:
                del config.final_config_dict['model_file']
            
            # Large dataset tuning
            if dataset in ['ml-20m', 'Electronics']:
                config['MAX_ITEM_LIST_LENGTH'] = 200
                config['train_batch_size'] = 512
            
            result = run_recbole(
                model=model,
                dataset=dataset,
                config_file_list=config_files[model],
                config_dict=config.final_config_dict,
                saved=True
            )
            
            with open(f'results/{model}_{dataset}.txt', 'w') as f:
                f.write(str(result))
            print(f"Completed {model} on {dataset}. Results: {result}")
        except Exception as e:
            print(f"Error running {model} on {dataset}: {str(e)}")
            traceback.print_exc()
            print("Skipping.")
