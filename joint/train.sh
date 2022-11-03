# CUDA_VISIBLE_DEVICES=3 python -u main_other.py --dataname TB-Dense --ignore_nonetype --epochs 50 --eval_steps 20 --log_steps 20
# CUDA_VISIBLE_DEVICES=0 python -u main_other.py --dataname TB-Dense --epochs 200 --eval_steps 20 --log_steps 20 --ignore_nonetype &
# CUDA_VISIBLE_DEVICES=1 python -u main_other.py --dataname MATRES --epochs 50 --eval_steps 100 --log_steps 50 --ignore_nonetype &


python -u main.py --eval_steps 200 --epochs 500 --lr 5e-4

# CUDA_VISIBLE_DEVICES=5 python -u main_other.py --dataname TB-Dense --epochs 500 --eval_steps 20 --log_steps 20 &

# CUDA_VISIBLE_DEVICES=7 python -u main_other.py --dataname MATRES --epochs 50 --eval_steps 100 --log_steps 50 &
# wait[]