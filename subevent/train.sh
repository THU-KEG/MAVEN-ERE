CUDA=$1
seed=$2

CUDA_VISIBLE_DEVICES=$CUDA python -u main.py \
    --eval_steps 50 --epochs 50 --batch_size 4 --seed $seed

CUDA_VISIBLE_DEVICES=$CUDA python -u main_other.py \
    --dataname hievents \
    --epochs 5000 --eval_steps 500 --log_steps 500 --batch_size 16 --seed $seed