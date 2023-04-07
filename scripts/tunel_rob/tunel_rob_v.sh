cd ../..
python -u main_informer.py --model ctt --task tunel_v --attn full --embed none --train_epochs 15 --lradj type4 --patience 5 --e_layers 3 --show_para 0 --seq_len 3 --seq_lenv 1 --batch_size 32