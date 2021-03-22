
# My gpu memory support two processes
max_children=4
function parallel {
  # Credits: Arnaldo Candido Junior
  local time1=$(date +"%H:%M:%S")
  local time2=""

  echo "starting $1 ($time1)..."
  "$@" && time2=$(date +"%H:%M:%S") && echo "finishing $2 ($time1 -- $time2)..." &

  local my_pid=$$
  local children=$(ps -eo ppid | grep -w $my_pid | wc -w)
  children=$((children-1))
  if [[ $children -ge $max_children ]]; then
    wait -n
  fi
}

PID=1314
while [ -e /proc/$PID ]
do
    sleep .6
done

PID=684
while [ -e /proc/$PID ]
do
    sleep .6
done

CUDA_VISIBLE_DEVICES=0 parallel python train.py -c Experiments/configs/Speech/exp1.json > ../Speech/Checkpoints/Experiment-1_train.log
CUDA_VISIBLE_DEVICES=1 parallel python train.py -c Experiments/configs/Speech/exp2.json > ../Speech/Checkpoints/Experiment-2_train.log
CUDA_VISIBLE_DEVICES=0 parallel python train.py -c Experiments/configs/Speech/exp3.json > ../Speech/Checkpoints/Experiment-3_train.log
CUDA_VISIBLE_DEVICES=1 parallel python train.py -c Experiments/configs/Speech/exp4.json > ../Speech/Checkpoints/Experiment-4_train.log
# CUDA_VISIBLE_DEVICES=1 parallel python train.py -c Experiments/configs/Speech/exp5.json > ../Speech/Checkpoints/Experiment-5_train.log
CUDA_VISIBLE_DEVICES=1 parallel  python train.py -c Experiments/configs/Speech/exp7.json > ../Speech/Checkpoints/Experiment-7_train.log
CUDA_VISIBLE_DEVICES=1 parallel  python train.py -c Experiments/configs/Speech/exp8.json > ../Speech/Checkpoints/Experiment-8_train.log

CUDA_VISIBLE_DEVICES=1  python train.py -c Experiments/configs/Speech/exp15.json > ../Speech/Checkpoints/Experiment-15_train.log

wait
