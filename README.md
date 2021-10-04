# Marginal Utility Diminishes in BERT KD

This repository contains the implementation of the [paper](https://arxiv.org/abs/2106.05691) "Marginal Utility Diminishes: Exploring the Minimum Knowledge for BERT Knowledge Distillation" in ACL 2021.

The code for fine-tuning models (w/o knowledge distillation (KD)) is modified from [huggingface/transformers](https://github.com/huggingface/transformers).

The code for TinyBERT is modified from [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT).

The code for ROSITA is modified from [ROSITA](https://github.com/llyx97/Rosita)

## Requirements

Python3 <br />
torch>1.4.0 <br />
tqdm <br />
boto3 <br />

## Fine-tune BERT Teacher
To fine-tune the pre-trained BERT model on a downstream task $TASK, run:
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
  --output_dir ${OUTPUT_DIR_FOR_STUDENT_MODEL_AFTER_PRED_DISTILL}$ \
  --max_seq_length 128 \
  --train_batch_size 64 \
  --num_train_epochs 5 \
  --learning_rate 2e-5 \
  --eval_step 50 \
  --do_lower_case \
  --pred_distill \
  --is_rosita
```
The above example is for ROSITA. To conduct distillation for TinyBERT, fill the argument `--student_model` with the path of the pre-trained TinyBERT, and delete the argument `--is_rosita`. The training hyper-parameters for each task can be found in the appendix of the paper. Notably, when the student is trained only with prediction (w/o HSK distillation), we set `--num_train_epochs` to the same as HSK distillation.

## Knowledge Distillation with Single-dimension HSK Compression

### Depth Compression
To conduct hidden state knowledge (HSK) distillation with HSK compressed from the depth dimension, run:
```
python main.py \
  --teacher_model models/bert_ft/$TASK \
  --student_model models/prun_bert/$TASK \
  --data_dir data/$TASK \
  --task_name $TASK \
  --output_dir ${OUTPUT_DIR_FOR_STUDENT_MODEL_AFTER_HSK_DISTILL}$ \
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
`--keep_layers` can be choosen from 1\~7 (including the embedding layer) for ROSITA and 1\~5 for TinyBERT. `--layer_scheme` can be choosen from "t_top8", "t_top10" and "t_top12".

### Length Compression
To conduct HSK distillation with HSK compressed from the length dimension, run:
```
python main.py \
  --teacher_model models/bert_ft/$TASK \
  --student_model models/prun_bert/$TASK \
  --data_dir data/$TASK \
  --task_name $TASK \
  --output_dir ${OUTPUT_DIR_FOR_STUDENT_MODEL_AFTER_HSK_DISTILL}$ \
  --train_batch_size 32 \
  --learning_rate 5e-5 \
  --eval_step 200 \
  --keep_tokens ${THE_NUM_OF_TOKENS_TO_KEEP_FOR_HSK}$  \
  --layer_scheme t_top10 \
  --token_scheme ${LENGTH_COMPRESSION_SCHEME}$ \
  --att_ttop12 False \
  --do_lower_case \
  --repr_distill \
  --is_rosita 
```
`--keep_tokens` can be set as any integer from 1 to maximum sequence length. `--token_scheme` can be choosen from "left_first", "attention" and "attention_no_sep". To enable selecting the attention scores using "t_top12", set `--att_ttop12` as True.

### Width Compression
A running example of HSK distillation with width compression is:
```
python3 main.py \
  --teacher_model models/bert_ft/$TASK \
  --student_model models/prun_bert/$TASK \
  --data_dir data/$TASK \
  --task_name $TASK \
  --output_dir ${OUTPUT_DIR_FOR_STUDENT_MODEL_AFTER_HSK_DISTILL}$ \
  --train_batch_size 32 \
  --learning_rate 5e-5 \
  --eval_step 200 \
  --keep_hidden ${THE_PERSENTAGE_OF_NEURONS_TO_KEEP_FOR_HSK}$ \
  --layer_scheme t_top10 \
  --hidden_scheme ${WIDTH_COMPRESSION_SCHEME}$ \
  --do_lower_case \
  --repr_distill \
  --is_rosita 
```
`--keep_hidden` can be set as any real number from 0\~1. `--hidden_scheme` can be choosen "rand_mask", "uniform_mask" and "importance_mask_dynamic".

### Prediction Distillation
After HSK distillation, the student model is further trained with prediction distillation:
```
python3 main.py \
  --teacher_model models/bert_ft/$TASK \
  --student_model ${OUTPUT_DIR_FOR_STUDENT_MODEL_AFTER_HSK_DISTILL}$ \
  --data_dir data/$TASK \
  --task_name $TASK \
  --output_dir ${OUTPUT_DIR_FOR_STUDENT_MODEL_AFTER_PRED_DISTILL}$ \
  --train_batch_size 32 \
  --num_train_epochs 5 \
  --warmup_proportion 0. \
  --learning_rate 2e-5 \
  --eval_step 50 \
  --do_lower_case \
  --pred_distill \
  --is_rosita
```

## Knowledge Distillation with Three-dimension HSK Compression
To conduct HSK distillation with three dimensions being compressed jointly, run:
```
python3 main.py \
  --teacher_model models/bert_ft/$TASK \
  --student_model models/prun_bert/$TASK \
  --data_dir data/$TASK \
  --task_name $TASK \
  --output_dir ${OUTPUT_DIR_FOR_STUDENT_MODEL_AFTER_HSK_DISTILL}$ \
  --config_3d 1l_10t_0.1h \
  --train_batch_size 32 \
  --learning_rate 5e-5 \
  --eval_step 200 \
  --do_lower_case \
  --repr_distill \
  --hidden_scheme importance_mask_dynamic \
  --token_scheme attention_no_sep \
  --layer_scheme t_top8 \
  --is_rosita
```
`--config_3d` specifies the configuration for HSK compression, i.e., the amount of HSK allocated to the depth, length and width dimensions. We adopt the "attention_no_sep" scheme for length compression, and the "importance_mask_dynamic" scheme for width compression, as they perform well in single dimension compression.
