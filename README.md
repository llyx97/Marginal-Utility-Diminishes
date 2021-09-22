# Marginal Utility Diminishes in BERT KD

This repository contains implementation of the [paper](https://arxiv.org/abs/2106.05691) "Marginal Utility Diminishes: Exploring the Minimum Knowledge for BERT Knowledge Distillation" in ACL 2021.

The code for fine-tuning models (w/o knowledge distillation (KD)) is modified from [huggingface/transformers](https://github.com/huggingface/transformers).

The code for TinyBERT is modified from [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT).

The code for ROSITA is modified from [ROSITA](https://github.com/llyx97/Rosita)

## Requirements

Python3 <br />
torch>1.4.0 <br />
tqdm <br />
boto3 <br />

## Fine-tune BERT Teacher
To fine-tune the pre-trained BERT model on a downstream task ${TASK_NAME}$, run:
```
CUDA_VISIBLE_DEVICES=0 python bert_ft.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --output_dir models/bert_ft/${TASK_NAME}$ \
  --data_dir data/${TASK_NAME}$ \
  --task_name ${TASK_NAME}$ \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --logging_steps 50 \
  --save_steps 0 
```

## Initialize Student Models

### TinyBERT 
The pre-trained TinyBERT can be downloaded from [this url](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT).

### ROSITA 
Step1: Compute the importance of model weights using a metric based on first-order taylor expansion. This can be achieved by running:
```
python bert_ft.py \
  --model_type bert \
  --model_name_or_path models/bert_ft/${TASK_NAME}$ \
  --output_dir models/bert_ft/${TASK_NAME}$/importance_score \
  --data_dir data/${TASK_NAME}$ \
  --task_name ${TASK_NAME}$ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --num_train_epochs 1.0 \
  --save_steps 0 \
  --compute_taylor True
```

Step2: Prune the model based on the importance scores:
```
python pruning.py \
   --model_path models/bert_ft/${TASK_NAME}$ \
   --output_dir models/prun_bert/${TASK_NAME}$ \
   --task ${TASK_NAME}$ \
   --keep_heads ${NUM_OF_ATTN_HEADS_TO_KEEP}$ \
   --num_layers ${NUM_OF_LAYERS_TO_KEEP}$ \
   --ffn_hidden_dim ${HIDDEN_DIM_OF_FFN}$ \
   --emb_hidden_dim ${MATRIX_RANK_OF_EMB_FACTORIZATION}$
```
The architecture of the ROSITA model is `keep_heads=2`, `keep_layers=6`, `ffn_hidden_dim=768` and `emb_hidden_dim=128`.
