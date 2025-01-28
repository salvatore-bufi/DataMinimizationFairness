from elliot.run import run_experiment
from runtimes_config.config_exp_template import TEMPLATE_BPR, TEMPLATE_EASER, TEMPLATE_LIGHTGCN, TEMPLATE_MULTIVAE, TEMPLATE_USERKNN
import os


CONFIG_DIR = './config_files_data_min'
log_file_path = os.path.abspath('./log_error_experiments.txt')

assert os.path.exists(CONFIG_DIR)
def train_models():
    DATASETS = ['ml-1m', 'ambar']
    TEMPLATES = {'BPR': TEMPLATE_BPR, 'EASER': TEMPLATE_EASER, 'LIGHT_GCN': TEMPLATE_LIGHTGCN,
                 'MULTIVAE': TEMPLATE_MULTIVAE, 'USER_KNN': TEMPLATE_USERKNN}
    for dataset in DATASETS:
        if dataset == 'ml-1m':
            STRATEGIES = ['highest_variance', 'least_favorite', 'most_characteristic', 'most_favorite',
                          'most_rated', 'most_recent', 'random', 'full']
        elif dataset == 'ambar':
            STRATEGIES = ['highest_variance', 'least_favorite', 'most_characteristic', 'most_favorite',
                          'most_rated', 'random', 'full']
        for model, TEMPLATE in TEMPLATES.items():
            for strategy in STRATEGIES:
                if strategy == 'full':
                    INTERACTIONS_NUM = ['1']
                else:
                    INTERACTIONS_NUM = ['1', '3', '7', '15', '100']
                for int_n in INTERACTIONS_NUM:
                    dataset_name = dataset + '_' + strategy + '_' + int_n
                    config = TEMPLATE.format(dataset=dataset, strategy=strategy, interactions_numb=int_n, dataset_name=dataset_name)
                    config_path = os.path.abspath(os.path.join(CONFIG_DIR, 'runtime_metrics_conf.yml'))
                    with open(config_path, 'w') as file:
                        file.write(config)
                    try:
                        run_experiment(config_path)
                    except Exception as e:
                        with open(log_file_path, 'a') as log_file:
                            error_msg = f'Error Processing {TEMPLATE} \n {strategy} \n {int_n} \n {model} \n {str(e)} \n \n '
                            log_file.write(error_msg)
