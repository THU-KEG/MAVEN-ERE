# CUDA_VISIBLE_DEVICES=4 python -u main_other.py --dataname MATRES --eval_only --load_ckpt ../output/MATRES_ignore_none_True/best --ignore_nonetype &
# CUDA_VISIBLE_DEVICES=4 python -u main_other.py --dataname MATRES --eval_only --load_ckpt ../output/MATRES_ignore_none_False/best &
# CUDA_VISIBLE_DEVICES=4 python -u main_other.py --dataname TB-Dense --eval_only --load_ckpt ../output/TB-Dense_ignore_none_True/best --ignore_nonetype &
# CUDA_VISIBLE_DEVICES=4 python -u main_other.py --dataname TB-Dense --eval_only --load_ckpt ../output/TB-Dense_ignore_none_False/best &
# CUDA_VISIBLE_DEVICES=4 python -u main.py --eval_only --ignore_nonetype &
# CUDA_VISIBLE_DEVICES=4 python -u main.py --eval_only &

CUDA_VISIBLE_DEVICES=4 python -u main_other.py --dataname TCR --eval_only --load_ckpt ../output/MATRES_ignore_none_False/best &

wait