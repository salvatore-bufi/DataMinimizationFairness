import json
from typing import Dict, List
from .save_recs_template import TEMPLATE_BPR, TEMPLATE_EASER, TEMPLATE_LIGHTGCN, TEMPLATE_MULTIVAE, TEMPLATE_USERKNN
import os


TEMPLATES = {'BPRMF': TEMPLATE_BPR, 'EASER': TEMPLATE_EASER, 'LightGCN': TEMPLATE_LIGHTGCN,
                             'MultiVAE': TEMPLATE_MULTIVAE, 'UserKNN': TEMPLATE_USERKNN}




def extract_models_parameter(model_parameter_json_path: str) -> Dict[str, str]:

    configuration = dict()
    with open(model_parameter_json_path) as file:
        config_data = json.load(file)

    for entry in config_data:
        if 'recommender' in entry:
            configuration['model'] = entry['recommender'].split('_')[0]
        if 'configuration' in entry:
            configuration.update(entry['configuration'])
    str_dict = {str(key): str(value) for key, value in configuration.items()}
    return str_dict

def extract_best_model_files_names(dataset_name: str) -> List[str]:
    performance_directory = os.path.abspath(os.path.join('./results', dataset_name, 'performance'))
    best_models_files_names_list = [file for file in os.listdir(performance_directory) if file.startswith("bestmodel")]
    return best_models_files_names_list

def extract_best_model_files_absolute_paths(dataset_name: str) -> List[str]:
    performance_directory = os.path.abspath(os.path.join('./results', dataset_name, 'performance'))
    best_models_files_absolute_paths_list = [os.path.join(performance_directory, file) for file in os.listdir(performance_directory) if file.startswith("bestmodel")]
    return best_models_files_absolute_paths_list

def fulfill_template(template: str, str_dict: Dict[str, str]) -> str:
    # Handle best_iteration overrides for validation_rate and epochs
    # str_dict contains the best model hyperparameter in the format: parameter_name : value
    if 'best_iteration' in str_dict:
        str_dict['validation_rate'] = str_dict['best_iteration']
        str_dict['epochs'] = str_dict['best_iteration']

    # Fill the TEMPLATE using the formatted dictionary
    try:
        fulfilled_template = template.format(**str_dict)
    except KeyError as e:
        missing_key = str(e).strip("'")
        raise ValueError(f"Missing required parameter in the dictionary: {missing_key}")

    return fulfilled_template


