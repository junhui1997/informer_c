cd ..
python -u main_informer.py --model ctt --task jigsaw_kt_gvk --attn prob --embed none --train_epochs 30 --lradj type3 --patience 5 --e_layers 3 --show_para 0 --seq_len 36 --distil --batch_size 128 --itr 2 --loss focal
python -u main_informer.py --model ctt --task jigsaw_kt_gvk --attn prob --embed none --train_epochs 30 --lradj type3 --patience 5 --e_layers 3 --show_para 0 --seq_len 48  --distil --batch_size 128 --itr 2 --loss focal
python -u main_informer.py --model ctt --task jigsaw_kt_gvk --attn prob --embed none --train_epochs 30 --lradj type3 --patience 5 --e_layers 3 --show_para 0 --seq_len 64 --distil --batch_size 128 --itr 2 --loss focal
python -u main_informer.py --model ctt --task jigsaw_kt_gvk --attn prob --embed none --train_epochs 30 --lradj type3 --patience 5 --e_layers 3 --show_para 0 --seq_len 81 --distil --batch_size 128 --itr 2 --loss focal
python -u main_informer.py --model ctt --task jigsaw_kt_gvk --attn full --embed none --train_epochs 30 --lradj type3 --patience 5 --e_layers 3 --show_para 0 --seq_len 36  --batch_size 128 --itr 2
python -u main_informer.py --model ctt --task jigsaw_kt_gvk --attn full --embed none --train_epochs 30 --lradj type3 --patience 5 --e_layers 3 --show_para 0 --seq_len 48  --batch_size 128 --itr 2
python -u main_informer.py --model ctt --task jigsaw_kt_gvk --attn full --embed none --train_epochs 30 --lradj type3 --patience 5 --e_layers 3 --show_para 0 --seq_len 64 --batch_size 128 --itr 2
python -u main_informer.py --model ctt --task jigsaw_kt_gvk --attn full --embed none --train_epochs 30 --lradj type3 --patience 5 --e_layers 3 --show_para 0 --seq_len 81 --batch_size 128 --itr 2

