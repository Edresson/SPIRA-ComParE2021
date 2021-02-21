

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

CUDA_VISIBLE_DEVICES=0 parallel python train.py -c Experiments/configs/Tosse/exp1.json > ../Tosse/Checkpoints/Experiment-1_train.log
CUDA_VISIBLE_DEVICES=1 parallel python train.py -c Experiments/configs/Tosse/exp2.json > ../Tosse/Checkpoints/Experiment-2_train.log
CUDA_VISIBLE_DEVICES=0 parallel python train.py -c Experiments/configs/Tosse/exp3.json > ../Tosse/Checkpoints/Experiment-3_train.log
CUDA_VISIBLE_DEVICES=1 parallel python train.py -c Experiments/configs/Tosse/exp4.json > ../Tosse/Checkpoints/Experiment-4_train.log
CUDA_VISIBLE_DEVICES=1  python train.py -c Experiments/configs/Tosse/exp5.json > ../Tosse/Checkpoints/Experiment-5_train.log
wait
