cd ..
python -u main_informer.py --model informer --task jigsaw_np --attn prob --embed none --train_epochs 30 --lradj type3 --patience 10 --e_layers 6 --itr 2
python -u main_informer.py --model informer --task jigsaw_su --attn prob --embed none --train_epochs 30 --lradj type3 --patience 10 --e_layers 6 --itr 2
python -u main_informer.py --model informer --task jigsaw_kt --attn prob --embed none --train_epochs 30 --lradj type3 --patience 10 --e_layers 6 --itr 2