cd ../..
python -u main_informer.py --model ctt_kv --task tunel_kv --attn full --embed none --train_epochs 30 --lradj type4 --patience 8 --e_layers 2 --show_para 0 --seq_len 36 --seq_lenv 4 --batch_size 16 --dual_img 1 --s 8 --itr 3 --learning_rate 0.00005 --num_workers 8



