SEED=111

CUDA_VISIBLE_DEVICES=0 python population/kgbert.py \
  --train_file ./data/train.csv \
  --encoder_name 'roberta-large' \
  --output_dir './models/kgbert_roberta_large_lr1e-5_bs64_epoch3_seed'$SEED \
  --per_device_train_batch_size 64 \
  --max_seq_length 32 \
  --learning_rate 1e-5 \
  --report_to 'none' \
  --save_strategy 'epoch' \
  --num_train_epochs 1 \
  --save_total_limit 1 \
  --seed $SEED \
  --do_train --do_eval --do_predict
