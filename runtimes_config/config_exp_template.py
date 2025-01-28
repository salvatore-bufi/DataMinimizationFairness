TEMPLATE_BPR = """experiment:
  backend: pytorch
  data_config:
    strategy: fixed
    train_path: ../dataset/{dataset}/{strategy}/{interactions_numb}.tsv
    validation_path: ../dataset/{dataset}/val.tsv
    test_path: ../dataset/{dataset}/test.tsv
  dataset: {dataset_name}
  top_k: 10
  evaluation:
    cutoffs: [10]
    simple_metrics: [nDCGRendle2020]
  gpu: 0
  external_models_path: ../external/models/__init__.py
  models:
    external.BPRMF:
      meta:
        verbose: True
        save_recs: False
        validation_rate: 1
        validation_metric: nDCGRendle2020@10
      lr: [0.01, 0.005, 0.001]
      epochs: 300
      factors: 64
      batch_size: 1024
      l_w: [0.01, 0.005, 0.001]
      seed: 123
      early_stopping:
        patience: 5
        mode: auto
        monitor: nDCGRendle2020@10
        verbose: True
"""



TEMPLATE_MULTIVAE = """experiment:
  backend: tensorflow
  data_config:
    strategy: fixed
    train_path: ../dataset/{dataset}/{strategy}/{interactions_numb}.tsv
    validation_path: ../dataset/{dataset}/val.tsv
    test_path: ../dataset/{dataset}/test.tsv
  dataset: {dataset_name}
  top_k: 10
  evaluation:
    cutoffs: [10]
    simple_metrics: [nDCGRendle2020]
  gpu: 0
  external_models_path: ../external/models/__init__.py
  models:
    MultiVAE:
      meta:
        verbose: True
        save_recs: False
        validation_rate: 1
        validation_metric: nDCGRendle2020@10
      lr: [0.01, 0.001]
      intermediate_dim: [100, 200]
      latent_dim: [100, 200]
      epochs: 300
      dropout_pkeep: 0.5
      reg_lambda: 0.005
      batch_size: 1024
      seed: 123
      early_stopping:
        patience: 5
        mode: auto
        monitor: nDCGRendle2020@10
        verbose: True
"""


TEMPLATE_LIGHTGCN = """experiment:
  backend: pytorch
  data_config:
    strategy: fixed
    train_path: ../dataset/{dataset}/{strategy}/{interactions_numb}.tsv
    validation_path: ../dataset/{dataset}/val.tsv
    test_path: ../dataset/{dataset}/test.tsv
  dataset: {dataset_name}
  top_k: 10
  evaluation:
    cutoffs: [10]
    simple_metrics: [nDCGRendle2020]
  gpu: -1
  external_models_path: ../external/models/__init__.py
  models:
    external.LightGCN:
      meta:
        verbose: True
        save_recs: False
        validation_rate: 1
        validation_metric: nDCGRendle2020@10
      lr: [0.01, 0.005, 0.001]
      epochs: 300
      factors: 64
      batch_size: 1024
      l_w: [0.01, 0.005, 0.001]
      n_layers: 2
      seed: 123
      early_stopping:
        patience: 5
        mode: auto
        monitor: nDCGRendle2020@10
        verbose: True
"""

TEMPLATE_EASER = """experiment:
  backend: tensorflow
  data_config:
    strategy: fixed
    train_path: ../dataset/{dataset}/{strategy}/{interactions_numb}.tsv
    validation_path: ../dataset/{dataset}/val.tsv
    test_path: ../dataset/{dataset}/test.tsv
  dataset: {dataset_name}
  top_k: 10
  evaluation:
    cutoffs: [10]
    simple_metrics: [nDCGRendle2020]
  gpu: 0
  external_models_path: ../external/models/__init__.py
  models:
    EASER:
      meta:
        verbose: True
        save_recs: False
        validation_rate: 1
        validation_metric: nDCGRendle2020@10
      l2_norm: [500, 1000, 5000]
      seed: 123
      early_stopping:
        patience: 5
        mode: auto
        monitor: nDCGRendle2020@10
        verbose: True
"""

TEMPLATE_USERKNN = """experiment:
  backend: tensorflow
  data_config:
    strategy: fixed
    train_path: ../dataset/{dataset}/{strategy}/{interactions_numb}.tsv
    validation_path: ../dataset/{dataset}/val.tsv
    test_path: ../dataset/{dataset}/test.tsv
  dataset: {dataset_name}
  top_k: 10
  evaluation:
    cutoffs: [10]
    simple_metrics: [nDCGRendle2020]
  gpu: 0
  external_models_path: ../external/models/__init__.py
  models:
    UserKNN:
      meta:
        verbose: True
        save_recs: False
        validation_rate: 1
        validation_metric: nDCGRendle2020@10
      neighbors: [20, 30, 50]
      similarity: [cosine, jaccard]
      seed: 123
      early_stopping:
        patience: 5
        mode: auto
        monitor: nDCGRendle2020@10
        verbose: True
"""

