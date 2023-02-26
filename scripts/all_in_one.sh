cd ..
python -u main_informer.py --model informer --task jigsaw_np --attn prob --embed none --train_epochs 30 --lradj type4 --patience 5 --e_layers 6 --show_para 0 --seq_len 64 --batch_size 128 --itr 3 --distil --loss focal
python -u main_informer.py --model informer --task jigsaw_su --attn prob --embed none --train_epochs 30 --lradj type4 --patience 5 --e_layers 6 --show_para 0 --seq_len 64 --batch_size 128 --itr 3 --distil --loss focal
python -u main_informer.py --model informer --task jigsaw_kt --attn prob --embed none --train_epochs 30 --lradj type4 --patience 5 --e_layers 6 --show_para 0 --seq_len 64 --batch_size 128 --itr 3 --distil --loss focal

python -u main_informer.py --model ctt --task jigsaw_kt_gv --attn full --embed none --train_epochs 30 --lradj type4 --patience 5 --e_layers 3 --show_para 0 --seq_len 8 --batch_size 128 --itr 3 --distil --loss focal
python -u main_informer.py --model ctt --task jigsaw_np_gv --attn full --embed none --train_epochs 30 --lradj type4 --patience 5 --e_layers 3 --show_para 0 --seq_len 8 --batch_size 128 --itr 3 --distil --loss focal
python -u main_informer.py --model ctt --task jigsaw_su_gv --attn full --embed none --train_epochs 30 --lradj type4 --patience 5 --e_layers 3 --show_para 0 --seq_len 8 --batch_size 128 --itr 3 --distil --loss focal



