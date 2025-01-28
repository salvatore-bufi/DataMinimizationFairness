import os.path

from elliot.run import run_experiment
from runtimes_config.compute_save_recs_template import TEMPLATES, extract_best_model_files_absolute_paths, extract_models_parameter, fulfill_template


def save_recs():
    DATASETS = ['ambar', 'ml-1m']
    CONFIG_DIR = './config_files'
    log_file_path = os.path.abspath('/log_error_recs.txt')
    config_path = os.path.abspath(os.path.join(CONFIG_DIR, 'runtime_compute_recs.yml'))
    for dataset in DATASETS:
        if dataset == 'ml-1m':
            STRATEGIES = ['highest_variance', 'least_favorite', 'most_characteristic', 'most_favorite',
                          'most_rated', 'most_recent', 'random', 'full']
        elif dataset == 'ambar':
            STRATEGIES = ['highest_variance', 'least_favorite', 'most_characteristic', 'most_favorite',
                          'most_rated', 'random', 'full']
        for strategy in STRATEGIES:
            if strategy == 'full':
                INTERACTIONS_NUM = ['1']
            else:
                INTERACTIONS_NUM = ['1', '3', '7', '15', '100']
            for int_n in INTERACTIONS_NUM:
                dataset_name = dataset + '_' + strategy + '_' + int_n
                print(f" Computing recs for {dataset_name} \n" )
                # list of best models files
                best_models_config_files_path_list = extract_best_model_files_absolute_paths(dataset_name=dataset_name)
                for best_model_path in best_models_config_files_path_list:
                    actual_configuration = extract_models_parameter(best_model_path)
                    current_model = actual_configuration['model']
                    print(f" \t\t Model: {current_model}")
                    current_template = TEMPLATES[current_model]
                    actual_configuration['dataset'] = dataset
                    actual_configuration['dataset_name'] = dataset_name
                    actual_configuration['strategy'] = strategy
                    actual_configuration['interactions_numb'] = int_n

                    current_config = fulfill_template(current_template, actual_configuration)
                    with open(config_path, 'w') as file:
                        file.write(current_config)
                    try:
                        run_experiment(config_path)
                    except Exception as e:
                        with open(log_file_path, 'a') as log_file:
                            error_msg = f'Error Processing {current_config} \n{strategy}, \n{int_n} \n{str(e)}\n\n'
                            log_file.write(error_msg)

