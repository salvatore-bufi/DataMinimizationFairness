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
        save_recs: True
        validation_rate: {validation_rate}
        validation_metric: nDCGRendle2020@10
      lr: {lr}
      epochs: {epochs}
      factors: 64
      batch_size: 1024
      l_w: {l_w}
      seed: 123
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
        save_recs: True
        validation_rate: {validation_rate}
        validation_metric: nDCGRendle2020@10
      lr: {lr}
      intermediate_dim: {intermediate_dim}
      latent_dim: {latent_dim}
      epochs: {epochs}
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
        save_recs: True
        validation_rate: {validation_rate}
        validation_metric: nDCGRendle2020@10
      lr: {lr}
      epochs: {epochs}
      factors: 64
      batch_size: 1024
      l_w: {l_w}
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
        save_recs: True
        validation_rate: 1
        validation_metric: nDCGRendle2020@10
      l2_norm: {l2_norm}
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
        save_recs: True
        validation_rate: 1
        validation_metric: nDCGRendle2020@10
      neighbors: {neighbors}
      similarity: {similarity}
      seed: 123
      early_stopping:
        patience: 5
        mode: auto
        monitor: nDCGRendle2020@10
        verbose: True
"""
