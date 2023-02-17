cd ..
python -u main_informer.py --model ctt --task jigsaw_kt_gv --attn prob --embed none --train_epochs 1 --lradj type3 --patience 10 --e_layers 2 --show_para 0 --seq_len 6 --itr 2
python -u main_informer.py --model ctt --task jigsaw_kt_gv --attn prob --embed none --train_epochs 1 --lradj type3 --patience 10 --e_layers 3 --show_para 0 --seq_len 6 --itr 2
python -u main_informer.py --model ctt --task jigsaw_kt_gv --attn prob --embed none --train_epochs 1 --lradj type3 --patience 10 --e_layers 4 --show_para 0 --seq_len 6 --itr 2