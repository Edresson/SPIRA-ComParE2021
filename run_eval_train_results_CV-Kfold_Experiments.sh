# Speech
echo "==================== Experiments with 5 Folds Eval - Train ===================="

# CUDA_VISIBLE_DEVICES=1 python test_eval_K-Fold_CV_models.py --experiment_dir ../Speech/Experiments_Final_kfolds/Experiment-1/ > ../Speech/Experiments_Final_kfolds/Experiment-1/cv_dev_Train_results.txt

# CUDA_VISIBLE_DEVICES=1 python test_eval_K-Fold_CV_models.py --experiment_dir ../Speech/Experiments_Final_kfolds/Experiment-2/ > ../Speech/Experiments_Final_kfolds/Experiment-2/cv_dev_Train_results.txt

# CUDA_VISIBLE_DEVICES=1 python test_eval_K-Fold_CV_models.py --experiment_dir ../Speech/Experiments_Final_kfolds/Experiment-3/ > ../Speech/Experiments_Final_kfolds/Experiment-3/cv_dev_Train_results.txt

# CUDA_VISIBLE_DEVICES=0 python test_eval_K-Fold_CV_models.py --experiment_dir ../Speech/Experiments_Final_kfolds/Experiment-4/ > ../Speech/Experiments_Final_kfolds/Experiment-4/cv_dev_Train_results.txt

# Cough

CUDA_VISIBLE_DEVICES=1 python test_eval_K-Fold_CV_models.py --experiment_dir ../Tosse/Experiments_Final_kfolds/Experiment-1/ > ../Tosse/Experiments_Final_kfolds/Experiment-1/cv_dev_Train_results.txt
CUDA_VISIBLE_DEVICES=1 python test_eval_K-Fold_CV_models.py --experiment_dir ../Tosse/Experiments_Final_kfolds/Experiment-2/ > ../Tosse/Experiments_Final_kfolds/Experiment-2/cv_dev_Train_results.txt
CUDA_VISIBLE_DEVICES=1 python test_eval_K-Fold_CV_models.py --experiment_dir ../Tosse/Experiments_Final_kfolds/Experiment-3/ > ../Tosse/Experiments_Final_kfolds/Experiment-3/cv_dev_Train_results.txt
CUDA_VISIBLE_DEVICES=1 python test_eval_K-Fold_CV_models.py --experiment_dir ../Tosse/Experiments_Final_kfolds/Experiment-4/ > ../Tosse/Experiments_Final_kfolds/Experiment-4/cv_dev_Train_results.txt
