from minimize_dataset import align_user_ratings_ambar, preprocess_dataset_ambar, apply_minimization_strategy, preprocess_dataset_ml
from prepare_side_information_for_complex_metrics import run_extraction_movielens_1_m, run_extraction_ambar
from train_models import train_models
from save_recs_for_best_models import save_recs
from compute_metrics import compute_metric

################### Create Minimized Datasets ###################
## AMBAR
print(f" Preprocessing Dataset: AMBAR")
align_user_ratings_ambar(dataset_name='ambar')
data_path = preprocess_dataset_ambar(dataset_name='ambar', dataset_inter_name='ratings_info.tsv', user_col='user_id', k_core=45, item_col='track_id', rating_col='rating')
run_extraction_ambar()
INTERACTIONS_N = [1, 3, 7, 15, 100]
print("Applying Minimization Strategies: Ambar Dataset")
for ni in INTERACTIONS_N:
    apply_minimization_strategy(df_path=data_path, dataset='ambar', df_name='dm_candidate.tsv',
                                val_path=None, test_path=None, strategy='random', n=ni, user_col_name='user_id', rating_col_name='rating', item_col_name='track_id')
    apply_minimization_strategy(df_path=data_path, dataset='ambar', df_name='dm_candidate.tsv',
                                val_path=None, test_path=None, strategy='most_favorite', n=ni,
                                user_col_name='user_id', rating_col_name='rating', item_col_name='track_id')
    apply_minimization_strategy(df_path=data_path, dataset='ambar', df_name='dm_candidate.tsv',
                                val_path=None, test_path=None, strategy='least_favorite', n=ni,
                                user_col_name='user_id', rating_col_name='rating', item_col_name='track_id')
    apply_minimization_strategy(df_path=data_path, dataset='ambar', df_name='dm_candidate.tsv',
                                val_path=None, test_path=None, strategy='most_rated', n=ni,
                                item_col_name='track_id', user_col_name='user_id', rating_col_name='rating')
    apply_minimization_strategy(df_path=data_path, dataset='ambar', df_name='dm_candidate.tsv',
                                val_path=None, test_path=None, strategy='most_characteristic', n=ni,
                                item_col_name='track_id', user_col_name='user_id', rating_col_name='rating')
    apply_minimization_strategy(df_path=data_path, dataset='ambar', df_name='dm_candidate.tsv',
                                val_path=None, test_path=None, strategy='highest_variance', n=ni,
                                item_col_name='track_id', user_col_name='user_id', rating_col_name='rating')
# FULL METHOD
apply_minimization_strategy(df_path=data_path, dataset='ambar', df_name='dm_candidate.tsv',
                                val_path=None, test_path=None, strategy='full', n=1, rating_col_name='rating', user_col_name='user_id', item_col_name='track_id')


## Movielens
print(f" Preprocessing Dataset: Movielens 1M")
data_path = preprocess_dataset_ml()
run_extraction_movielens_1_m()
print("Applying Minimization Strategies: Movielens 1M Dataset")
for n in INTERACTIONS_N:

    apply_minimization_strategy(df_path=data_path, dataset='ml-1m', df_name='dm_candidate.tsv',
                                val_path=None, test_path=None, strategy='random', n=n, user_col_name='user_id:token', item_col_name='item_id:token',rating_col_name='rating:float')
    apply_minimization_strategy(df_path=data_path, dataset='ml-1m', df_name='dm_candidate.tsv',
                                val_path=None, test_path=None, strategy='most_recent', n=n,
                                user_col_name='user_id:token', timestamp_col_name='timestamp:float', item_col_name='item_id:token',rating_col_name='rating:float')
    apply_minimization_strategy(df_path=data_path, dataset='ml-1m', df_name='dm_candidate.tsv',
                                val_path=None, test_path=None, strategy='most_favorite', n=n,
                                user_col_name='user_id:token', rating_col_name='rating:float', item_col_name='item_id:token')
    apply_minimization_strategy(df_path=data_path, dataset='ml-1m', df_name='dm_candidate.tsv',
                                val_path=None, test_path=None, strategy='least_favorite', n=n,
                                user_col_name='user_id:token', rating_col_name='rating:float', item_col_name='item_id:token')
    apply_minimization_strategy(df_path=data_path, dataset='ml-1m', df_name='dm_candidate.tsv',
                                val_path=None, test_path=None, strategy='most_rated', n=n,
                                item_col_name='item_id:token', user_col_name='user_id:token', rating_col_name='rating:float')
    apply_minimization_strategy(df_path=data_path, dataset='ml-1m', df_name='dm_candidate.tsv',
                                val_path=None, test_path=None, strategy='most_characteristic', n=n,
                                item_col_name='item_id:token', user_col_name='user_id:token', rating_col_name='rating:float')
    apply_minimization_strategy(df_path=data_path, dataset='ml-1m', df_name='dm_candidate.tsv',
                                val_path=None, test_path=None, strategy='highest_variance', n=n,
                                item_col_name='item_id:token', user_col_name='user_id:token',
                                rating_col_name='rating:float')

apply_minimization_strategy(df_path=data_path, dataset='ml-1m', df_name='dm_candidate.tsv',
                                val_path=None, test_path=None, strategy='full', n=1, user_col_name='user_id:token', item_col_name='item_id:token',rating_col_name='rating:float')



######################## Train Models
train_models()

######################## Save RECS for Best Models
save_recs()


######################## Compute Metrics on Best Models
compute_metric()