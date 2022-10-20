CUDA=$1
seed=$2

CUDA_VISIBLE_DEVICES=$CUDA python -u main_other.py \
    --dataname CausalTimeBank \
    --epochs 200 --eval_steps 100 --log_steps 50 --K 10 --seed $seed 
    
CUDA_VISIBLE_DEVICES=$CUDA python -u main_other.py \
    --dataname EventStoryLine \
    --epochs 50 --eval_steps 100 --log_steps 50 --K 5 --seed $seed

CUDA_VISIBLE_DEVICES=$CUDA python -u main.py \
    --eval_steps 500 --epochs 50 --batch_size 4 --seed $seed