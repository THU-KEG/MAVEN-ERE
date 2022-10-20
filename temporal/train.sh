# run on MAVEN under setting 1
python -u main.py --epochs 50 --log_steps 20 --eval_steps 50
# run on MAVEN under setting 2
python -u main.py --epochs 50 --log_steps 20 --eval_steps 50 --ignore_nonetype

# run on MATRES, TB-Dense under setting 2
python -u main_other.py --dataname MATRES --epochs 50 --eval_steps 100 --log_steps 50 --ignore_nonetype
python -u main_other.py --dataname TB-Dense --epochs 500 --eval_steps 20 --log_steps 20 --ignore_nonetype