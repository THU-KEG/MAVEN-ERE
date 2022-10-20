CUDA=$1
seed=$2

CUDA_VISIBLE_DEVICES=$CUDA python -u main.py \
    --dataset maven \
    --epochs 20 --eval_steps 20 --log_steps 20 --seed $seed 

CUDA_VISIBLE_DEVICES=$CUDA python -u main.py \
    --dataset ace \
    --epochs 20 --eval_steps 20 --log_steps 20 --seed $seed

CUDA_VISIBLE_DEVICES=$CUDA python -u main.py \
    --dataset kbp \
    --epochs 20 --eval_steps 50 --log_steps 50 --seed $seed