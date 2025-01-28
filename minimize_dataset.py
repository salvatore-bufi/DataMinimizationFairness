import os
import pandas as pd
from typing import List
import shutil

from data_minimization import data_splitting
from data_minimization import minimization_strategies

INTERACTIONS_N = [1, 3, 7, 15, 100]

STRATEGIES = ['full', 'random', 'most_recent', 'most_favorite', 'least_favorite', 'most_rated', 'most_characteristic',
              'highest_variance']

function_mapping = {
    'full': minimization_strategies.full_min,
    'random': minimization_strategies.random_min,
    'most_recent': minimization_strategies.most_recent_min,
    'most_favorite': minimization_strategies.most_favorite_min,
    'least_favorite': minimization_strategies.least_favorite_min,
    'most_rated': minimization_strategies.most_rated_min,
    'most_characteristic': minimization_strategies.most_characteristic_min,
    'highest_variance': minimization_strategies.highest_variance_min
}


def check_k_core(df: pd.DataFrame, column_name: str = 'user_id:token'):
    val_count = df[column_name].value_counts()
    min_count = val_count.min()
    min_value = val_count.idxmin()
    print(min_count, min_value)


def create_directory(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created : {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")
    return None


def copy_and_rename(source_file_path: str, dest_file_path: str) -> None:
    if not os.path.isfile(source_file_path):
        raise FileNotFoundError(f" File {source_file_path} not found !")
    try:
        shutil.copy2(source_file_path, dest_file_path)
        print(f" Copied {source_file_path} \t {dest_file_path}")
    except IOError as e:
        print(f" Unable to copy file. {e}")
    except Exception as e:
        print(f"Unexpected error {e}")


def apply_minimization_strategy(df_path: str, dataset: str = 'ml-1m', df_name: str = 'dm_candidate.tsv',
                                strategy: str = 'full', val_path: str = None, test_path: str = None, **kwargs) -> str:
    strategy = strategy.lower()
    if strategy not in STRATEGIES:
        all_strategies_names = "\n".join(["\t-" + n for n in STRATEGIES])
        raise ValueError(
            f" Strategy: {strategy} not implemented.\n The implemented strategies are:\n {all_strategies_names}")

    df = pd.read_csv(os.path.join(df_path, df_name), sep='\t')
    try:
        func = function_mapping[strategy]
        minimized_df = func(df, **kwargs)
    except TypeError as e:
        raise TypeError(f" Error calling function '{strategy}': {e}")

    # new_df_name = dataset + '-' + strategy
    new_df_path_method = os.path.abspath(os.path.join('./dataset', dataset, strategy))
    create_directory(new_df_path_method)  # directory for the method minimization

    train_file_name = f"{kwargs['n']}" + '.tsv'
    # train_file_name = f"{n}" + '.tsv'

    minimized_df.to_csv(os.path.join(new_df_path_method, train_file_name + 'headers_full'), sep='\t', index=False)

    user_col_name = kwargs['user_col_name']
    item_col_name = kwargs['item_col_name']
    rating_col_name = kwargs['rating_col_name']
    # minimized_df_elliot = minimized_df[['user_id:token', 'item_id:token', 'rating:float']].copy()
    minimized_df_elliot = minimized_df[[user_col_name, item_col_name, rating_col_name]].copy()
    minimized_df_elliot.to_csv(os.path.join(new_df_path_method, train_file_name), sep='\t', index=False, header=False)

    path = df_path

    dest_val_path = os.path.abspath(os.path.join('./dataset', dataset, 'val.tsv'))
    if not os.path.isfile(dest_val_path):
        if val_path is None:
            val_path = os.path.join(path, 'dm_val.tsv')  # actual
            copy_and_rename(val_path, dest_val_path)

    dest_test_path = os.path.abspath(os.path.join('./dataset', dataset, 'test.tsv'))
    if test_path is None:
        test_path = os.path.join(path, 'dm_test.tsv')  # actual
        copy_and_rename(test_path, dest_test_path)
    return 'miao'

def preprocess_dataset_ambar(dataset_name: str = 'ambar', dataset_inter_name: str = 'ratings_info.tsv',
                       user_col: str = 'user_id', k_core: int = 45,
                       user_based_split: float = 0.7, subsample: int = 2500, item_col: str = 'track_id', rating_col:str = 'rating') -> str:
    data_directory = './data'
    dataset_path = os.path.abspath(os.path.join(data_directory, dataset_name, dataset_inter_name))

    # Check wheter the file exist and it is in the correct directory
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"The dataset file '{dataset_inter_name}' was not found at the path: {dataset_path} \n"
                                f"files must be positioned in {data_directory} \\ dataset_name")
    print(f"File {dataset_path} found. \n Begin user k-core {45} ")
    df = pd.read_csv(dataset_path, sep='\t')


    df_cored = data_splitting.iterative_k_core(df, column1=user_col, column2=item_col, k=k_core) # used on ambar
    df_subsampled = data_splitting.subsample_by_column(df=df_cored, column=user_col, n=subsample, seed=42)

    print(f"k-core finished. Begin user-based split in ds: {70}% and dm: {30}% \n")
    # user_based split - ds: 70% of users, dm: 30% of users and saving:
    ds, dm = data_splitting.user_based_split(df_subsampled, user_col=user_col, percentage=user_based_split, seed=42)
    # create data_path
    current_ds_name = 'user_based'
    current_data_path = os.path.abspath(
        os.path.join(data_directory, dataset_name, dataset_name + '-' + current_ds_name))
    create_directory(current_data_path)
    ds.to_csv(os.path.join(current_data_path, 'ds.tsv'), sep='\t', index=False)
    dm.to_csv(os.path.join(current_data_path, 'dm.tsv'), sep='\t', index=False)

    print(f" User-splitting finished. File saved in {current_data_path} \n Begin data splitting 70-10-20")
    # Splitting DM
    dm_candidate, dm_val, dm_test = data_splitting.split_dataset_per_user(dm, user_col=user_col, train_ratio=0.7,
                                                                          val_ratio=0.1, test_ratio=0.2)
    dm_candidate.to_csv(os.path.join(current_data_path, 'dm_candidate.tsv'), sep='\t', index=False)


    dm_val_elliot = dm_val[[user_col, item_col, rating_col]].copy()
    dm_val_elliot.to_csv(os.path.join(current_data_path, 'dm_val.tsv'), sep='\t', index=False, header=False)
    dm_val.to_csv(os.path.join(current_data_path, 'dm_val_full.tsv'), sep='\t', index=False)

    dm_test.to_csv(os.path.join(current_data_path, 'dm_test_full.tsv'), sep='\t', index=False)
    dm_test_elliot = dm_test[[user_col, item_col, rating_col]].copy()
    dm_test_elliot.to_csv(os.path.join(current_data_path, 'dm_test.tsv'), sep='\t', index=False, header=False)


    print(f" Data-splitting finished. Files saved in {current_data_path} \n")
    return current_data_path


def preprocess_dataset_ml(dataset_name: str = 'ml-1m', dataset_inter_name: str = 'inter.tsv',
                       user_col: str = 'user_id:token', k_core: int = 45,
                       user_based_split: float = 0.7, subsample: int = 2500, item_col: str = 'item_id:token', rating_col:str = 'rating:float',
                       strategies: List[str] = 'full') -> str:
    data_directory = './data'
    dataset_path = os.path.abspath(os.path.join(data_directory, dataset_name, dataset_inter_name))


    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"The dataset file '{dataset_inter_name}' was not found at the path: {dataset_path} \n"
                                f"files must be positioned in {data_directory} \\ dataset_name")
    print(f"File {dataset_path} found. \n Begin user k-core {k_core} ")
    df = pd.read_csv(dataset_path, sep='\t')


    df_cored = data_splitting.k_core(df, column=user_col, k=k_core)
    df_subsampled = data_splitting.subsample_by_column(df=df_cored, column=user_col, n=subsample, seed=42)

    print(f"k-core finished. Begin user-based split in ds: {70}% and dm: {30}% \n")
    # user_based split - ds: 70% of users, dm: 30% of users and saving:
    ds, dm = data_splitting.user_based_split(df_subsampled, user_col=user_col, percentage=user_based_split, seed=42)
    # create data_path
    current_ds_name = 'user_based'
    current_data_path = os.path.abspath(
        os.path.join(data_directory, dataset_name, dataset_name + '-' + current_ds_name))
    create_directory(current_data_path)
    ds.to_csv(os.path.join(current_data_path, 'ds.tsv'), sep='\t', index=False)
    dm.to_csv(os.path.join(current_data_path, 'dm.tsv'), sep='\t', index=False)

    print(f" User-splitting finished. File saved in {current_data_path} \n Begin data splitting 70-10-20")
    # Splitting DM
    dm_candidate, dm_val, dm_test = data_splitting.split_dataset_per_user(dm, user_col=user_col, train_ratio=0.7,
                                                                          val_ratio=0.1, test_ratio=0.2)
    dm_candidate.to_csv(os.path.join(current_data_path, 'dm_candidate.tsv'), sep='\t', index=False)


    dm_val_elliot = dm_val[[user_col, item_col, rating_col]].copy()
    dm_val_elliot.to_csv(os.path.join(current_data_path, 'dm_val.tsv'), sep='\t', index=False, header=False)
    dm_val.to_csv(os.path.join(current_data_path, 'dm_val_full.tsv'), sep='\t', index=False)

    dm_test.to_csv(os.path.join(current_data_path, 'dm_test_full.tsv'), sep='\t', index=False)
    dm_test_elliot = dm_test[[user_col, item_col, rating_col]].copy()
    dm_test_elliot.to_csv(os.path.join(current_data_path, 'dm_test.tsv'), sep='\t', index=False, header=False)


    print(f" Data-splitting finished. Files saved in {current_data_path} \n")
    return current_data_path


def align_user_ratings_ambar(dataset_name: str = 'ambar'):
    ambar_path = os.path.abspath(os.path.join('./data', dataset_name))
    files_names = ['artists_info', 'ratings_info', 'tracks_info', 'users_info']

    for file_name in files_names:
        current_file = os.path.join(ambar_path, file_name)
        df = pd.read_csv(current_file + '.csv', sep=',')
        df.to_csv(current_file + '.tsv', sep='\t', index=False)

    # removing users without continent information

    # Load the user_info.tsv file
    users_info_path = os.path.join(ambar_path, 'users_info.tsv')

    users_info = pd.read_csv(users_info_path, sep='\t')

    # Load the rating_info.tsv file
    ratings_info_path = os.path.join(ambar_path, 'ratings_info.tsv')
    ratings_info = pd.read_csv(ratings_info_path, sep='\t')

    # Remove entries in users_info where continent is NaN
    users_info_cleaned = users_info.dropna(subset=['continent'])

    # Get the list of user_ids that remain
    valid_user_ids = set(users_info_cleaned['user_id'])

    # Remove entries in rating_info where user_id is not in the valid_user_ids
    ratings_info_cleaned = ratings_info[ratings_info['user_id'].isin(valid_user_ids)]

    # Rename Uncleaned file:
    ratings_info_path_renamed = os.path.join(ambar_path, 'ratings_info_uncleaned.tsv')
    os.rename(ratings_info_path, ratings_info_path_renamed)

    users_info_path_renamed = os.path.join(ambar_path, 'users_info_uncleaned.tsv')
    os.rename(users_info_path, users_info_path_renamed)

    # Save the cleaned data back to files
    users_info_cleaned.to_csv(users_info_path, sep='\t', index=False)
    ratings_info_cleaned.to_csv(ratings_info_path, sep='\t', index=False)