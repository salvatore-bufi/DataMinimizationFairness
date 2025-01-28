METRICS_TEMPLATE = """experiment:
  backend: pytorch
  data_config:
    strategy: fixed
    train_path: ../dataset/{dataset}/{strategy}/{interactions_numb}.tsv
    validation_path: ../dataset/{dataset}/val.tsv
    test_path: ../dataset/{dataset}/test.tsv
  dataset: {dataset_name}
  top_k: 20
  evaluation:
    cutoffs: [1, 10, 20]
    paired_ttest: True
    simple_metrics: [nDCG, nDCGRendle2020, HR, Recall, PopREO, PopRSP, Gini, ItemCoverage]
    complex_metrics:
     - metric: REO
       clustering_name: {item_clustering_name}
       clustering_file: ../data/{dataset}/{item_clustering_file_name}
     - metric: RSP
       clustering_name: {item_clustering_name}
       clustering_file: ../data/{dataset}/{item_clustering_file_name}
     - metric: BiasDisparityBD
       user_clustering_name: {user_clustering_name}
       user_clustering_file: ../data/{dataset}/{user_clustering_file_name}
       item_clustering_name: {item_clustering_name}
       item_clustering_file: ../data/{dataset}/{item_clustering_file_name}
     - metric: BiasDisparityBR
       user_clustering_name: {user_clustering_name}
       user_clustering_file: ../data/{dataset}/{user_clustering_file_name}
       item_clustering_name: {item_clustering_name}
       item_clustering_file: ../data/{dataset}/{item_clustering_file_name}
     - metric: BiasDisparityBS
       user_clustering_name: {user_clustering_name}
       user_clustering_file: ../data/{dataset}/{user_clustering_file_name}
       item_clustering_name: {item_clustering_name}
       item_clustering_file: ../data/{dataset}/{item_clustering_file_name}
     - metric: UserMADRanking
       clustering_name: {user_clustering_name}
       clustering_file: ../data/{dataset}/{user_clustering_file_name}
  gpu: 0
  models:
    RecommendationFolder:  
        folder: results/{dataset_name}/recs
"""