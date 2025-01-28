import os
import pandas as pd
from typing import List

# from save_recs_for_best_models import dataset_name
import pandas as pd


def load_and_merge_data(dataset_name: str = 'ambar', type: str = 'item', file_1_name='tracks_info.tsv', file_2_name='artists_info.tsv',
                        key: str = 'artist_id') -> [str, str]:
    """
    Reads track and artist information from TSV files, cleans the data by removing specified columns,
    and merges the two datasets on the artist_id.

    Parameters:
    - file_1 (str): Name to the file_1 TSV file ( where to add information from file_2).
    - file_2 (str): Name to the file_2 TSV file (where to take information to add to file_1).

    Returns:
    - pd.DataFrame: Merged DataFrame with columns ['track_id', 'artist_id', 'gender', 'country', 'continent'].
    """

    path = os.path.abspath(os.path.join('./data', dataset_name))

    file_1_path = os.path.join(path, file_1_name)
    file_2_path = os.path.join(path, file_2_name)

    # file_1 = pd.read_csv(file_1_path, sep='\t')
    file_2 = pd.read_csv(file_2_path, sep='\t')
    # Read the track_info.tsv file into a DataFrame
    try:
        file_1 = pd.read_csv(file_1_path, sep='\t')
        print(f"Loaded {file_1_path} successfully.")
    except FileNotFoundError:
        print(f"Error: The file {file_1_path} was not found.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Could not parse {file_1_path}. Please check the file format.")
        return None

    # Define columns to drop from track_df if they exist
    file_1_cols_to_drop = ['category_styles', 'styles', 'duration']
    # Drop the columns, ignoring any that don't exist
    file_1.drop(columns=[col for col in file_1_cols_to_drop if col in file_1.columns], inplace=True)
    print(f"Dropped columns {file_1_cols_to_drop} from {file_1_path} if they existed.")

    # Read the artists_info.tsv file into a DataFrame
    try:
        file_2 = pd.read_csv(file_2_path, sep='\t')
        print(f"Loaded {file_2_path} successfully.")
    except FileNotFoundError:
        print(f"Error: The file {file_2_path} was not found.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Could not parse {file_2_path}. Please check the file format.")
        return None

    # Define columns to drop from artists_df if they exist
    file_2_cols_to_drop = ['category_styles', 'styles', 'duration']
    # Drop the columns, ignoring any that don't exist
    file_2.drop(columns=[col for col in file_2_cols_to_drop if col in file_2.columns], inplace=True)
    print(f"Dropped columns {file_2_cols_to_drop} from {file_2_path} if they existed.")

    # Merge the two DataFrames on 'artist_id' using a left join to retain all tracks
    merged_df = pd.merge(file_1, file_2, on=key, how='left')
    print(f"Merged {file_1_name} and {file_2_name} DataFrames on {key}.")

    # Select the desired columns for the final DataFrame
    final_columns = ['track_id', 'artist_id', 'gender', 'country', 'continent']

    # Check if all final columns exist in the merged DataFrame
    missing_columns = [col for col in final_columns if col not in merged_df.columns]
    if missing_columns:
        print(f"Warning: The following expected columns are missing from the merged DataFrame: {missing_columns}")
        # You can choose to handle this case as needed, e.g., fill with NaN or raise an error
        for col in missing_columns:
            merged_df[col] = pd.NA  # Assign NaN for missing columns

    # Create the final DataFrame with the selected columns
    final_df = merged_df[final_columns]
    print("Selected the final columns for the merged DataFrame.")

    # Save path
    df_save_name = type + '_full_info' + '.tsv'
    path_to_save = os.path.join(path, df_save_name)
    final_df.to_csv(path_to_save, sep='\t', index=False)
    print(f"Full information dataset saved in {path_to_save}")
    return df_save_name, path_to_save


def remap_column_with_group_mapping(dataframe: pd.DataFrame, column_entity: [int, str], column_attribute: [int, str],
                                    path: str, n_groups: int = None, type: str = '') -> pd.DataFrame:
    """
    Remap the values in a specified column of a dataframe to continuous integers starting from zero
    or group them into specified ranges.

    Save the mapping to a file.

    Parameters:
        dataframe (pd.DataFrame): The input dataframe.
        column_entity (int or str): The column containing either user or item ids, specified by name or column index.
        column_attribute (int or str): The column to remap or group, specified by name or column index.
        path (str): The path to save the mapping file and the extracted dataframe.
        n_groups (int, optional): Number of groups to divide numerical column values into. If not specified, perform a 1-to-1 mapping.

    Returns:
        pd.DataFrame: The dataframe with the remapped column.
    """
    # Handle dataframes without headers
    if not isinstance(dataframe.columns, pd.Index):
        dataframe.columns = range(dataframe.shape[1])

    # Get column name if column is specified by index
    if isinstance(column_attribute, int):
        attribute_column_name = dataframe.columns[column_attribute]
        entity_column_name = dataframe.columns[column_entity]
    else:
        attribute_column_name = column_attribute
        entity_column_name = column_entity

    if n_groups is not None:
        # Group numerical values into ranges
        min_value = dataframe[attribute_column_name].min()
        max_value = dataframe[attribute_column_name].max()
        range_step = (max_value - min_value) / n_groups

        # Create a mapping based on ranges
        def map_to_group(value):
            return int((value - min_value) // range_step) if value < max_value else n_groups - 1

        dataframe[attribute_column_name] = dataframe[attribute_column_name].apply(map_to_group)
        value_to_int = {f"Group {i}": i for i in range(n_groups)}
    else:
        print(f" Group number can not be None")

    # Save the mapping to a file
    mapping_file_name = type + '_' + str(attribute_column_name) + '_mapping.tsv'
    mapping_file_path = os.path.join(path, mapping_file_name)
    with open(mapping_file_path, 'w') as f:
        f.write("Original\tMapped\n")
        if n_groups is not None:
            for i in range(n_groups):
                f.write(f"Group {i}\t{i}\n")
        else:
            for original, mapped in value_to_int.items():
                f.write(f"{original}\t{mapped}\n")

    reduced_remapped_df = dataframe[[entity_column_name, attribute_column_name]]

    # Save reduced side information dataset for elliot
    dataframe_file_name = type + '_' + str(attribute_column_name) + '.tsv'
    dataframe_file_path = os.path.join(path, dataframe_file_name)
    reduced_remapped_df.to_csv(dataframe_file_path, sep='\t', header=False, index=False)
    return dataframe_file_path


def remap_column_with_mapping(dataframe: pd.DataFrame, column_entity: [int, str], column_attribute: [int, str],
                              path: str, type: str = '') -> pd.DataFrame:
    """
    Remap the values in a specified column of a dataframe to continuous integers starting from zero.
    Save the mapping to a file.

    Parameters:
        dataframe (pd.DataFrame): The input dataframe.
        column_entity(int or str): The column containing either user or item ids, specified by name or column index.
        column_attribute (int or str): The column to remap, specified by name or column index.
        path (str): The path to save the mapping file and the extracted dataframe
        type (str): Specify user or item to rename the saved dataframe

    Returns:
        pd.DataFrame: The dataframe with the remapped column.
    """
    # Handle dataframes without headers
    if not isinstance(dataframe.columns, pd.Index):
        dataframe.columns = range(dataframe.shape[1])

    # Get column name if column is specified by index
    if isinstance(column_attribute, int):
        attribute_column_name = dataframe.columns[column_attribute]
        entity_column_name = dataframe.columns[column_entity]
    else:
        attribute_column_name = column_attribute
        entity_column_name = column_entity

    # Get unique values and create a mapping
    unique_values = dataframe[attribute_column_name].unique()
    value_to_int = {value: idx for idx, value in enumerate(unique_values)}

    # Map the column values to integers
    dataframe[attribute_column_name] = dataframe[attribute_column_name].map(value_to_int)

    # Save the mapping to a file
    mapping_file_name = type + '_' + str(attribute_column_name) + '_mapping.tsv'
    mapping_file_path = os.path.join(path, mapping_file_name)
    with open(mapping_file_path, 'w') as f:
        f.write("Original\tMapped\n")
        for original, mapped in value_to_int.items():
            f.write(f"{original}\t{mapped}\n")

    reduced_remapped_df = dataframe[[entity_column_name, attribute_column_name]]

    # Save reduced side information dataset for elliot
    dataframe_file_name = type + '_' + str(attribute_column_name) + '.tsv'
    dataframe_file_path = os.path.join(path, dataframe_file_name)
    reduced_remapped_df.to_csv(dataframe_file_path, sep='\t', header=False, index=False)
    return dataframe_file_path


def extract_side_elliot_information(dataset_name: str, attribute_file_name: str, column_entity_name: [str, int],
                                    column_attribute_name: [str, int] = [], column_attribute_name_group: List
                                    = [], type: str = '', n_groups: int = None) -> None:
    path = os.path.abspath(os.path.join('./data', dataset_name))
    dataset_path = os.path.join(path, attribute_file_name)
    df = pd.read_csv(dataset_path, sep='\t')

    for att_name in column_attribute_name:
        saved_path = remap_column_with_mapping(dataframe=df, column_entity=column_entity_name,
                                               column_attribute=att_name, path=path, type=type)
        print(f"Extracted dataset saved in {saved_path}")
    for att_name in column_attribute_name_group:
        saved_path = remap_column_with_group_mapping(dataframe=df, column_entity=column_entity_name,
                                                     column_attribute=att_name, path=path, n_groups=n_groups, type=type)
        print(f"Extracted dataset saved in {saved_path}")

    print(f" Attribute for file {attribute_file_name} for dataset {dataset_name}  for type: {type} done! \n")
    return None


def run_extraction_movielens_1_m():
    dataset_name = 'ml-1m'

    type = 'item'
    attribute_file_name = 'item.tsv'
    column_entity_name = 'item_id:token'
    column_attribute_name_group = ['release_year:token']
    extract_side_elliot_information(dataset_name=dataset_name, attribute_file_name=attribute_file_name,
                                    column_entity_name=column_entity_name,
                                    column_attribute_name_group=column_attribute_name_group, type=type, n_groups=4)

    type = 'user'
    attribute_file_name = 'user.tsv'
    column_entity_name = 'user_id:token'
    column_attribute_name = ['gender:token']
    extract_side_elliot_information(dataset_name=dataset_name, attribute_file_name=attribute_file_name,
                                    column_entity_name=column_entity_name,
                                    column_attribute_name=column_attribute_name, type=type)


def run_extraction_ambar():
    dataset_name = 'ambar'


    type = 'item'
    attribute_file_name, _ = load_and_merge_data()
    column_entity_name = 'track_id'
    column_attribute_name = ['gender', 'country', 'continent']
    extract_side_elliot_information(dataset_name=dataset_name, attribute_file_name=attribute_file_name,
                                    column_entity_name=column_entity_name,
                                    column_attribute_name=column_attribute_name, type=type)

    type = 'user'
    attribute_file_name = 'users_info.tsv'
    column_entity_name = 'user_id'
    column_attribute_name = ['gender', 'country', 'continent']
    extract_side_elliot_information(dataset_name=dataset_name, attribute_file_name=attribute_file_name,
                                    column_entity_name=column_entity_name,
                                    column_attribute_name=column_attribute_name, type=type)