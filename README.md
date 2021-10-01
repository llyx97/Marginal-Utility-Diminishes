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
  --output_dir models/bert_ft/$TASK \
  --data_dir data/$TASK \
  --task_name $TASK \
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
  --model_name_or_path models/bert_ft/$TASK \
  --output_dir models/bert_ft/$TASK/importance_score \
  --data_dir data/$TASK \
  --task_name $TASK \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --num_train_epochs 1.0 \
  --save_steps 0 \
  --compute_taylor True
```

Step2: Prune the model based on the importance scores:
```
python pruning.py \
   --model_path models/bert_ft/$TASK \
   --output_dir models/prun_bert/$TASK \
   --task $TASK \
   --keep_heads ${NUM_OF_ATTN_HEADS_TO_KEEP}$ \
   --num_layers ${NUM_OF_LAYERS_TO_KEEP}$ \
   --ffn_hidden_dim ${HIDDEN_DIM_OF_FFN}$ \
   --emb_hidden_dim ${MATRIX_RANK_OF_EMB_FACTORIZATION}$
```
The architecture of the ROSITA model is `keep_heads=2`, `keep_layers=6`, `ffn_hidden_dim=768` and `emb_hidden_dim=128`.

## Knowledge Distillation with Teacher Predictions (Soft-labels)
To conduct prediction distillation, run
```
python main.py \
  --teacher_model models/bert_ft/$TASK \
  --student_model models/prun_bert/$TASK \
  --data_dir data/$TASK \
  --task_name $TASK \
  --output_dir ${OUTPUT_DIR_FOR_STUDENT_MODEL}$ \
  --max_seq_length 128 \
  --train_batch_size 64 \
  --num_train_epochs 5 \
  --learning_rate 2e-5 \
  --eval_step 50 \
  --do_lower_case \
  --pred_distill \
  --is_rosita
```
The above example is for ROSITA. To conduct distillation for TinyBERT, fill the argument `--student_model` with the path of the pre-trained TinyBERT, and delete the argument `--is_rosita`. The training hyper-parameters for each task can be found in the appendix of the paper. Notably, when the student is trianed only with prediction (w/o HSK distillation), we set the number of training epoch to the same as HSK distillation.

## Knowledge Distillation with Single-dimension HSK Compression

### Depth Compression
To conduct hidden state konwledge (HSK) distillation with HSK compressed from the depth dimension, run:
```
python main.py \
  --teacher_model models/bert_ft/$TASK \
  --student_model models/prun_bert/$TASK \
  --data_dir data/$TASK \
  --task_name $TASK \
  --output_dir ${OUTPUT_DIR_FOR_STUDENT_MODEL_AFTER_HSK_DISTILLATION}$ \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 5e-5 \
  --eval_step 200 \
  --keep_layers ${THE_NUM_OF_LAYERS_TO_KEEP_FOR_HSK}$ \
  --layer_scheme ${DEPTH_COMPRESSION_SCHEME}$ \
  --do_lower_case \
  --repr_distill \
  --is_rosita
```
