lr=5e-5
epochs=50

python -m eventualization.few_shot_evt \
  --train_file ./data/deco/deco_train.json \
  --none_sample_file ./data/none_sample_5each.csv \
  --model_name_or_path 't5-base' \
  --output_dir 'models/t5_12relation_lr'$lr'_epoch'$epochs'_seed42'  \
  --per_device_train_batch_size 4 \
  --report_to 'none' \
  --learning_rate $lr \
  --num_train_epochs $epochs \
  --save_total_limit 1 \
  --save_steps 500000000 \
  --overwrite_output_dir \
  --seed 42 \
  --do_train