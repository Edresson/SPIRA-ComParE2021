# Speech
# ok
CUDA_VISIBLE_DEVICES=1 nohup python train_kfold_5seeds.py -c  Experiments_Final_kfold/configs/Speech/exp1.json -f  ../Speech/dist/lab/5-fold/ > ../Speech/Experiments_Final_kfolds/Experiment-1/train-log.txt &
CUDA_VISIBLE_DEVICES=1 nohup python train_kfold_5seeds.py -c  Experiments_Final_kfold/configs/Speech/exp2.json -f  ../Speech/dist/lab/5-fold/ > ../Speech/Experiments_Final_kfolds/Experiment-2/train-log.txt &
CUDA_VISIBLE_DEVICES=1 nohup python train_kfold_5seeds.py -c  Experiments_Final_kfold/configs/Speech/exp3.json -f  ../Speech/dist/lab/5-fold/ > ../Speech/Experiments_Final_kfolds/Experiment-3/train-log.txt &
CUDA_VISIBLE_DEVICES=1 nohup python train_kfold_5seeds.py -c  Experiments_Final_kfold/configs/Speech/exp4.json -f  ../Speech/dist/lab/5-fold/ > ../Speech/Experiments_Final_kfolds/Experiment-4/train-log.txt &

# Cough
# wait to run
CUDA_VISIBLE_DEVICES=1 nohup python train_kfold_5seeds.py -c  Experiments_Final_kfold/configs/Tosse/exp1.json -f  ../Tosse/dist/lab/5-fold/ > ../Tosse/Experiments_Final_kfolds/Experiment-1/train-log.txt &
CUDA_VISIBLE_DEVICES=1 nohup python train_kfold_5seeds.py -c  Experiments_Final_kfold/configs/Tosse/exp2.json -f  ../Tosse/dist/lab/5-fold/ > ../Tosse/Experiments_Final_kfolds/Experiment-2/train-log.txt &
CUDA_VISIBLE_DEVICES=1 nohup python train_kfold_5seeds.py -c  Experiments_Final_kfold/configs/Tosse/exp3.json -f  ../Tosse/dist/lab/5-fold/ > ../Tosse/Experiments_Final_kfolds/Experiment-3/train-log.txt &
CUDA_VISIBLE_DEVICES=1 nohup python train_kfold_5seeds.py -c  Experiments_Final_kfold/configs/Tosse/exp4.json -f  ../Tosse/dist/lab/5-fold/ > ../Tosse/Experiments_Final_kfolds/Experiment-4/train-log.txt &



