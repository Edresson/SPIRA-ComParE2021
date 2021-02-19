#python train.py -c Experiments/configs/exp3.3.json &
#python train.py -c Experiments/configs/exp3.4.json

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

parallel CUDA_VISIBLE_DEVICES=0 python train.py -c Experiments/configs/exp1.json > ../Speech/Checkpoints/Experiment-1/train.log
parallel CUDA_VISIBLE_DEVICES=0 python train.py -c Experiments/configs/exp2.json > ../Speech/Checkpoints/Experiment-2/train.log
parallel CUDA_VISIBLE_DEVICES=1 python train.py -c Experiments/configs/exp3.json > ../Speech/Checkpoints/Experiment-3/train.log
parallel CUDA_VISIBLE_DEVICES=1 python train.py -c Experiments/configs/exp4.json > ../Speech/Checkpoints/Experiment-4/train.log
wait
