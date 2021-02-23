<< MULTILINE-COMMENT
# evaluation
aaa
MULTILINE-COMMENT
echo "==================== Experiment 1 ===================="
echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Tosse/dist/lab/devel.csv -r ../Tosse/dist/wav --checkpoint_path ../Tosse/Checkpoints/Experiment-1/spiraconv_v2/best_checkpoint.pt --config_path ../Tosse/Checkpoints/Experiment-1/spiraconv_v2/config.json  --batch_size 16 --num_workers 2 
echo "========================================================="


echo "==================== Experiment 2 ===================="
echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Tosse/dist/lab/devel.csv -r ../Tosse/dist/wav --checkpoint_path ../Tosse/Checkpoints/Experiment-2/spiraconv_v2/best_checkpoint.pt --config_path ../Tosse/Checkpoints/Experiment-2/spiraconv_v2/config.json  --batch_size 16 --num_workers 2 
echo "========================================================="


echo "==================== Experiment 3 ===================="
echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Tosse/dist/lab/devel.csv -r ../Tosse/dist/wav --checkpoint_path ../Tosse/Checkpoints/Experiment-3/spiraconv_v2/best_checkpoint.pt --config_path ../Tosse/Checkpoints/Experiment-3/spiraconv_v2/config.json  --batch_size 16 --num_workers 2 
echo "========================================================="


echo "==================== Experiment 4 ===================="
echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Tosse/dist/lab/devel.csv -r ../Tosse/dist/wav --checkpoint_path ../Tosse/Checkpoints/Experiment-4/spiraconv_v2/best_checkpoint.pt --config_path ../Tosse/Checkpoints/Experiment-4/spiraconv_v2/config.json  --batch_size 16 --num_workers 2 
echo "========================================================="

echo "==================== Experiment 5 ===================="
echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Tosse/dist/lab/devel.csv -r ../Tosse/dist/wav --checkpoint_path ../Tosse/Checkpoints/Experiment-5/spiraconv_v2/best_checkpoint.pt --config_path ../Tosse/Checkpoints/Experiment-5/spiraconv_v2/config.json  --batch_size 16 --num_workers 2 
echo "========================================================="


echo "==================== Experiment 6 ===================="
echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Tosse/dist/lab/devel.csv -r ../Tosse/dist/wav --checkpoint_path ../Tosse/Checkpoints/Experiment-5/spiraconv_v2/best_checkpoint.pt --config_path ../Tosse/Checkpoints/Experiment-6/spiraconv_v2/config.json  --batch_size 16 --num_workers 2 
echo "========================================================="
