cd ..
python -u main_informer.py --model informer --task navi_rob --attn prob --embed none --train_epochs 30 --lradj type3 --patience 10 --itr 1 --e_layers 2 --seq_len 128
python -u main_informer.py --model informer --task navi_rob --attn prob --embed none --train_epochs 30 --lradj type3 --patience 10 --itr 1 --e_layers 4 --seq_len 128
python -u main_informer.py --model informer --task navi_rob --attn prob --embed none --train_epochs 30 --lradj type3 --patience 10 --itr 1 --e_layers 6 --seq_len 128
python -u main_informer.py --model informer --task navi_rob --attn prob --embed none --train_epochs 30 --lradj type3 --patience 10 --itr 1 --e_layers 8 --seq_len 128