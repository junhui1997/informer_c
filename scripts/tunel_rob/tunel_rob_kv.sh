cd ../..
python -u main_informer.py --model ctt_kv --task tunel_kv --attn full --embed none --train_epochs 20 --lradj type4 --patience 5 --e_layers 3 --show_para 0 --seq_len 36 --seq_lenv 3 --batch_size 24 --dual_img 1 --s 6 --itr 3
python -u main_informer.py --model ctt_kv --task tunel_kv --attn full --embed none --train_epochs 20 --lradj type4 --patience 5 --e_layers 3 --show_para 0 --seq_len 36 --seq_lenv 3 --batch_size 24 --dual_img 1 --s 8 --itr 3
python -u main_informer.py --model ctt_kv --task tunel_kv --attn full --embed none --train_epochs 20 --lradj type4 --patience 5 --e_layers 3 --show_para 0 --seq_len 36 --seq_lenv 3 --batch_size 24 --dual_img 1 --s 10 --itr 3

python -u main_informer.py --model ctt_kv --task tunel_kv --attn full --embed none --train_epochs 20 --lradj type4 --patience 5 --e_layers 3 --show_para 0 --seq_len 36 --seq_lenv 4 --batch_size 16 --dual_img 1 --s 6 --itr 3
python -u main_informer.py --model ctt_kv --task tunel_kv --attn full --embed none --train_epochs 20 --lradj type4 --patience 5 --e_layers 3 --show_para 0 --seq_len 36 --seq_lenv 4 --batch_size 16--dual_img 1 --s 8 --itr 3
python -u main_informer.py --model ctt_kv --task tunel_kv --attn full --embed none --train_epochs 20 --lradj type4 --patience 5 --e_layers 3 --show_para 0 --seq_len 36 --seq_lenv 4 --batch_size 16 --dual_img 1 --s 10 --itr 3