cd ..
python -u main_informer.py --model ctt_gv --task jigsaw_kt_gvk --attn full --embed none --train_epochs 30 --lradj type4 --patience 5 --e_layers 3 --show_para 0 --seq_len 64 --batch_size 128 --itr 1 --distil --loss focal --do_predict 1
python -u main_informer.py --model ctt_gv --task jigsaw_np_gvk --attn full --embed none --train_epochs 30 --lradj type4 --patience 5 --e_layers 3 --show_para 0 --seq_len 64 --batch_size 128 --itr 1 --distil --loss focal --do_predict 1
python -u main_informer.py --model ctt_gv --task jigsaw_su_gvk --attn full --embed none --train_epochs 30 --lradj type4 --patience 5 --e_layers 3 --show_para 0 --seq_len 64 --batch_size 128 --itr 1 --distil --loss focal --do_predict 1

