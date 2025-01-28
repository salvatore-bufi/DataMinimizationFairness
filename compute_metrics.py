from runtimes_config.metrics_template import METRICS_TEMPLATE
import os.path
from elliot.run import run_experiment

def compute_metric():
    CONFIG_DIR = './config_files'
    log_file_path = os.path.abspath('/log_error_metrics.txt')
    config_path = os.path.abspath(os.path.join(CONFIG_DIR, 'runtime_compute_recs.yml'))

    DATASETS = ['ml-1m', 'ambar']
    for dataset in DATASETS:
        if dataset == 'ml-1m':
            STRATEGIES = ['highest_variance', 'least_favorite', 'most_characteristic', 'most_favorite',
                          'most_rated', 'most_recent', 'random', 'full']
            metrics_parameters = {'ml-1m': {'item_clustering_file_name': 'item_release_year:token.tsv',
                                            'item_clustering_name': 'item_release_year',
                                            'user_clustering_file_name': 'user_gender:token.tsv',
                                            'user_clustering_name': 'user_gender'}}
        elif dataset == 'ambar':
            STRATEGIES = ['highest_variance', 'least_favorite', 'most_characteristic', 'most_favorite',
                          'most_rated', 'random', 'full']
            metrics_parameters = {'ambar': {'item_clustering_file_name': 'item_continent.tsv',
                                            'item_clustering_name': 'item_continent',
                                            'user_clustering_file_name': 'user_continent.tsv',
                                            'user_clustering_name': 'user_continent'}}
        else:
            raise NotImplementedError
        current_parameter = metrics_parameters[dataset]
        current_parameter['dataset'] = dataset
        for strategy in STRATEGIES:
            if strategy == 'full':
                INTERACTIONS_NUM = ['1']
            else:
                INTERACTIONS_NUM = ['1', '3', '7', '15', '100']
            current_parameter['strategy']  = strategy
            for int_n in INTERACTIONS_NUM:
                current_parameter['interactions_numb'] = int_n

                dataset_name = dataset + '_' + strategy + '_' + int_n
                current_parameter['dataset_name'] = dataset_name
                print(f" Computing metrics for {dataset_name} \n" )

                try:
                    current_template = METRICS_TEMPLATE.format(**current_parameter)
                except KeyError as e:
                    missing_key = str(e).strip("'")
                    raise ValueError(f"Missing required parameter in the dictionary: {missing_key}")

                with open(config_path, 'w') as file:
                    file.write(current_template)

                run_experiment(config_path)