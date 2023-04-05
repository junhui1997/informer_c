cd ../..
python -u main_informer.py --model ctt_kv --task tunel_kv --attn full --embed none --train_epochs 15 --lradj type4 --patience 5 --e_layers 3 --show_para 0 --seq_len 20 --batch_size 32