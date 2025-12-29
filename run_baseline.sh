python3 src/main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food
python3 src/main.py --model_name LightGCN --emb_size 64 --n_layers 3 --lr 1e-3 --l2 1e-8 --dataset Grocery_and_Gourmet_Food
python3 src/main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 5e-4 --l2 1e-7 --dropout 0.2 --dataset Grocery_and_Gourmet_Food

python3 src/main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset ML_1MTOPK
python3 src/main.py --model_name LightGCN --emb_size 64 --n_layers 3 --lr 1e-3 --l2 1e-8 --dataset ML_1MTOPK
python3 src/main.py --model_name NeuMF --emb_size 64 --layers '[64]' --lr 5e-4 --l2 1e-7 --dropout 0.2 --dataset ML_1MTOPK
