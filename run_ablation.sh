python3 src/main.py --model_name AblationStagewiseGCN --emb_size 64 --lr 1e-4 --l2 0 --dataset Grocery_and_Gourmet_Food --c 0.1 --even_layer 4 --odd_layer 3 --epoch 500 --enabled_stage 1
python3 src/main.py --model_name AblationStagewiseGCN --emb_size 64 --lr 1e-4 --l2 0 --dataset Grocery_and_Gourmet_Food --c 0.1 --even_layer 4 --odd_layer 3 --epoch 500 --enabled_stage 2
python3 src/main.py --model_name AblationStagewiseGCN --emb_size 64 --lr 1e-4 --l2 0 --dataset Grocery_and_Gourmet_Food --c 0.1 --even_layer 4 --odd_layer 3 --epoch 500 --enabled_stage 3

python3 src/main.py --model_name AblationStagewiseGCN --emb_size 64 --lr 1e-4 --l2 0 --dataset ML_1MTOPK --c 0.1 --even_layer 4 --odd_layer 3 --epoch 500 --enabled_stage 1
python3 src/main.py --model_name AblationStagewiseGCN --emb_size 64 --lr 1e-4 --l2 0 --dataset ML_1MTOPK --c 0.1 --even_layer 4 --odd_layer 3 --epoch 500 --enabled_stage 2
python3 src/main.py --model_name AblationStagewiseGCN --emb_size 64 --lr 1e-4 --l2 0 --dataset ML_1MTOPK --c 0.1 --even_layer 4 --odd_layer 3 --epoch 500 --enabled_stage 3
