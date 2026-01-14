from recbole.quick_start import load_data_and_model
from recbole.trainer import Trainer
from recbole.config import Config
from recbole.utils import init_seed, init_logger, get_model, get_trainer
import os
import traceback

models = ['GRU4Rec', 'SASRec']
datasets = ['ml-1m', 'ml-20m', 'Books', 'Sports_and_Outdoors', 'Beauty', 'Electronics', 'yelp']

config_files = {
    'GRU4Rec': ['configs/general_config.yaml', 'configs/gru4rec_config.yaml'],
    'SASRec': ['configs/general_config.yaml', 'configs/sasrec_config.yaml']
}

os.makedirs('results', exist_ok=True)

for model_name in models:
    for dataset_name in datasets:
        dataset_dir = f"./data/{dataset_name}"
        inter_file = f"{dataset_dir}/{dataset_name}.inter"
        
        if not (os.path.exists(dataset_dir) and os.path.exists(inter_file)):
            print(f"Skipping {model_name} on {dataset_name}: Dataset folder or .inter file not found.")
            continue
        
        print(f"Running {model_name} on {dataset_name}...")
        
        try:
            # Load config
            config = Config(model=model_name, dataset=dataset_name, config_file_list=config_files[model_name])
            
            # Force single process (extra safety)
            config['nproc'] = 1
            config['world_size'] = 1
            config['local_rank'] = 0
            
            # Large dataset overrides
            if dataset_name in ['ml-20m', 'Electronics']:
                config['MAX_ITEM_LIST_LENGTH'] = 200
                config['train_batch_size'] = 512
            
            # Initialize seed and logger
            init_seed(config['seed'], config['reproducibility'])
            init_logger(config)
            
            # Load data and model (this is the safe, low-level way)
            dataset, train_data, valid_data, test_data = load_data_and_model(config)
            model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
            
            # Trainer
            trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
            
            # Train + evaluate
            best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=True)
            test_result = trainer.evaluate(test_data, load_best_model=True)
            
            result = {
                'best_valid_score': best_valid_score,
                'best_valid_result': best_valid_result,
                'test_result': test_result
            }
            
            with open(f'results/{model_name}_{dataset_name}.txt', 'w') as f:
                f.write(str(result))
            print(f"Completed {model_name} on {dataset_name}. Results: {result}")
        except Exception as e:
            print(f"Error running {model_name} on {dataset_name}: {str(e)}")
            traceback.print_exc()
            print("Skipping.")
