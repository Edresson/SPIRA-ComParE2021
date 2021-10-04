echo "==================== Experiments with 5 seeds ===================="

echo "==================== Experiment 1 ===================="
echo "____________________EVALUATION____________________"
python test_all_seeds_or_folds.py  --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav_normalized --experiment_dir ../Speech/Experiments_Final_one_window_Final/Experiment-1/
echo "--------------------------------------------------"
echo "______________________TRAIN_______________________"
python test_all_seeds_or_folds.py  --test_csv ../Speech/dist/lab/train.csv -r ../Speech/dist/wav_normalized --experiment_dir ../Speech/Experiments_Final_one_window_Final/Experiment-1/
echo "--------------------------------------------------"
echo "========================================================="

echo "==================== Experiment 2 ===================="
echo "____________________EVALUATION____________________"
CUDA_VISIBLE_DEVICES=1 python test_all_seeds_or_folds.py  --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav_normalized --experiment_dir ../Speech/Experiments_Final_one_window_Final/Experiment-2/
echo "--------------------------------------------------"
echo "______________________TRAIN_______________________"
CUDA_VISIBLE_DEVICES=1 python test_all_seeds_or_folds.py  --test_csv ../Speech/dist/lab/train.csv -r ../Speech/dist/wav_normalized --experiment_dir ../Speech/Experiments_Final_one_window_Final/Experiment-2/
echo "--------------------------------------------------"
echo "========================================================="

echo "==================== Experiment 3 ===================="
echo "____________________EVALUATION____________________"
CUDA_VISIBLE_DEVICES=1 python test_all_seeds_or_folds.py  --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav_normalized --experiment_dir ../Speech/Experiments_Final_one_window_Final/Experiment-3/
echo "--------------------------------------------------"
echo "______________________TRAIN_______________________"
CUDA_VISIBLE_DEVICES=1 python test_all_seeds_or_folds.py  --test_csv ../Speech/dist/lab/train.csv -r ../Speech/dist/wav_normalized --experiment_dir ../Speech/Experiments_Final_one_window_Final/Experiment-3/
echo "--------------------------------------------------"
echo "========================================================="

echo "==================== Experiment 4 ===================="
echo "____________________EVALUATION____________________"
CUDA_VISIBLE_DEVICES=1 python test_all_seeds_or_folds.py  --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav_normalized --experiment_dir ../Speech/Experiments_Final_one_window_Final/Experiment-4/
echo "--------------------------------------------------"
echo "______________________TRAIN_______________________"
CUDA_VISIBLE_DEVICES=1 python test_all_seeds_or_folds.py  --test_csv ../Speech/dist/lab/train.csv -r ../Speech/dist/wav_normalized --experiment_dir ../Speech/Experiments_Final_one_window_Final/Experiment-4/
echo "--------------------------------------------------"
echo "========================================================="


# nohup sh run_all_tests_Speech_3_second_window_experiments_seed.sh > ../Speech/Experiments_Final_one_window_Final/eval_results.txt &

# exemplo para o test
# CUDA_VISIBLE_DEVICES=1 python test_all_seeds_or_folds.py  --test_csv ../Speech/dist/lab/test.csv -r ../Speech/dist/wav_normalized --experiment_dir ../Speech/Experiments_Final_one_window_Final/Experiment-4/ --debug 0
# CUDA_VISIBLE_DEVICES=1 python test_all_seeds_or_folds.py  --test_csv ../Speech/dist/lab/test.csv -r ../Speech/dist/wav_normalized --experiment_dir ../Speech/Experiments_Final_one_window_Final/Experiment-4/ --debug 0