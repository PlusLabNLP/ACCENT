SEED=111
LR=1e-7

CUDA_VISIBLE_DEVICES=0 python helper/train_comet.py \
  --train_file data/train.csv \
  --model_name_or_path 'facebook/bart-large' \
  --output_dir './models/my_cskb_population_comet \
  --per_device_train_batch_size 64 \
  --learning_rate $LR \
  --report_to 'none' \
  --save_strategy 'epoch' \
  --seed $SEED \
  --num_train_epochs 1 \
  --do_train
