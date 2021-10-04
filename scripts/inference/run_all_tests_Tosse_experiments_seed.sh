echo "==================== Experiments with 5 seeds ===================="

echo "==================== Experiment 1 ===================="
echo "____________________EVALUATION____________________"
python test_all_seeds_or_folds.py  --test_csv ../Tosse/dist/lab/devel.csv -r ../Tosse/dist/wav_normalized --experiment_dir ../Tosse/Experiments_Final/Experiment-1/
echo "--------------------------------------------------"
echo "______________________TRAIN_______________________"
python test_all_seeds_or_folds.py  --test_csv ../Tosse/dist/lab/train.csv -r ../Tosse/dist/wav_normalized --experiment_dir ../Tosse/Experiments_Final/Experiment-1/
echo "--------------------------------------------------"
echo "========================================================="

echo "==================== Experiment 2 ===================="
echo "____________________EVALUATION____________________"
python test_all_seeds_or_folds.py  --test_csv ../Tosse/dist/lab/devel.csv -r ../Tosse/dist/wav_normalized --experiment_dir ../Tosse/Experiments_Final/Experiment-2/
echo "--------------------------------------------------"
echo "______________________TRAIN_______________________"
python test_all_seeds_or_folds.py  --test_csv ../Tosse/dist/lab/train.csv -r ../Tosse/dist/wav_normalized --experiment_dir ../Tosse/Experiments_Final/Experiment-2/
echo "--------------------------------------------------"
echo "========================================================="

echo "==================== Experiment 3 ===================="
echo "____________________EVALUATION____________________"
python test_all_seeds_or_folds.py  --test_csv ../Tosse/dist/lab/devel.csv -r ../Tosse/dist/wav_normalized --experiment_dir ../Tosse/Experiments_Final/Experiment-3/
echo "--------------------------------------------------"
echo "______________________TRAIN_______________________"
python test_all_seeds_or_folds.py  --test_csv ../Tosse/dist/lab/train.csv -r ../Tosse/dist/wav_normalized --experiment_dir ../Tosse/Experiments_Final/Experiment-3/
echo "--------------------------------------------------"
echo "========================================================="

echo "==================== Experiment 4 ===================="
echo "____________________EVALUATION____________________"
python test_all_seeds_or_folds.py  --test_csv ../Tosse/dist/lab/devel.csv -r ../Tosse/dist/wav_normalized --experiment_dir ../Tosse/Experiments_Final/Experiment-4/
echo "--------------------------------------------------"
echo "______________________TRAIN_______________________"
python test_all_seeds_or_folds.py  --test_csv ../Tosse/dist/lab/train.csv -r ../Tosse/dist/wav_normalized --experiment_dir ../Tosse/Experiments_Final/Experiment-4/
echo "--------------------------------------------------"
echo "========================================================="