SEED=42

CUDA_VISIBLE_DEVICES=0 python population/bert.py \
  --train_file ./data/train.csv \
  --encoder_name 'roberta-large' \
  --output_dir './models/bert_roberta_large_lr1e-5_bs64_epoch1_seed'$SEED \
  --per_device_train_batch_size 64 \
  --learning_rate 1e-5 \
  --report_to 'none' \
  --save_strategy 'epoch' \
  --save_total_limit 1 \
  --num_train_epochs 1 \
  --seed $SEED \
  --do_train --do_eval --do_predict
