<< MULTILINE-COMMENT
# evaluation
aaa

echo "==================== Experiment 1 ===================="
echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav --checkpoint_path ../Speech/Checkpoints/Experiment-1/spiraconv_v2/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-1/spiraconv_v2/config.json  --batch_size 16 --num_workers 2 
echo "========================================================="


echo "==================== Experiment 2 ===================="
echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav --checkpoint_path ../Speech/Checkpoints/Experiment-2/spiraconv_v2/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-2/spiraconv_v2/config.json  --batch_size 16 --num_workers 2 
echo "========================================================="


echo "==================== Experiment 3 ===================="
echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav --checkpoint_path ../Speech/Checkpoints/Experiment-3/spiraconv_v2/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-3/spiraconv_v2/config.json  --batch_size 16 --num_workers 2 
echo "========================================================="


echo "==================== Experiment 4 ===================="
echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav --checkpoint_path ../Speech/Checkpoints/Experiment-4/spiraconv_v2/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-4/spiraconv_v2/config.json  --batch_size 16 --num_workers 2 
echo "========================================================="


echo "==================== Experiment 5 ===================="
echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav --checkpoint_path ../Speech/Checkpoints/Experiment-5/spiraconv_v2/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-5/spiraconv_v2/config.json  --batch_size 16 --num_workers 2 
echo "========================================================="

echo "==================== Experiment 7 ===================="
echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav_normalized --checkpoint_path ../Speech/Checkpoints/Experiment-7/spt_v1/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-7/spt_v1/config.json  --batch_size 16 --num_workers 2 
echo "========================================================="




echo "==================== Experiment 22 ===================="
echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav_denoise --checkpoint_path ../Speech/Checkpoints/Experiment-22/spiraconv_v2/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-22/spiraconv_v2/config.json  --batch_size 16 --num_workers 2 
echo "____________________Train____________________"
python test.py --test_csv ../Speech/dist/lab/train.csv -r ../Speech/dist/wav_denoise --checkpoint_path ../Speech/Checkpoints/Experiment-22/spiraconv_v2/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-22/spiraconv_v2/config.json  --batch_size 16 --num_workers 2 

echo "========================================================="


echo "==================== Experiment 23 ===================="
echo "____________________Train____________________"
python test.py --test_csv ../Speech/dist/lab/train.csv -r ../Speech/dist/wav_so-denoise --checkpoint_path ../Speech/Checkpoints/Experiment-23/spiraconv_v2/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-23/spiraconv_v2/config.json  --batch_size 16 --num_workers 2 

echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav_so-denoise --checkpoint_path ../Speech/Checkpoints/Experiment-23/spiraconv_v2/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-23/spiraconv_v2/config.json  --batch_size 16 --num_workers 2 

echo "========================================================="

echo "==================== Experiment 24 ===================="
echo "____________________Train____________________"
python test.py --test_csv ../Speech/dist/lab/train.csv -r ../Speech/dist/wav_denoise --checkpoint_path ../Speech/Checkpoints/Experiment-24/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-24/*/config.json  --batch_size 16 --num_workers 2 

echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav_denoise --checkpoint_path ../Speech/Checkpoints/Experiment-24/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-24/*/config.json  --batch_size 16 --num_workers 2 

echo "========================================================="

echo "==================== Experiment 25 ===================="
echo "____________________Train____________________"
python test.py --test_csv ../Speech/dist/lab/train.csv -r ../Speech/dist/wav_so-denoise --checkpoint_path ../Speech/Checkpoints/Experiment-25/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-25/*/config.json  --batch_size 16 --num_workers 2 

echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav_so-denoise --checkpoint_path ../Speech/Checkpoints/Experiment-25/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-25/*/config.json  --batch_size 16 --num_workers 2 

echo "========================================================="

echo "==================== Experiment 26 ===================="
echo "____________________Train____________________"
python test.py --test_csv ../Speech/dist/lab/train.csv -r ../Speech/dist/wav_so-denoise --checkpoint_path ../Speech/Checkpoints/Experiment-26/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-26/*/config.json  --batch_size 16 --num_workers 2 

echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav_so-denoise --checkpoint_path ../Speech/Checkpoints/Experiment-26/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-26/*/config.json  --batch_size 16 --num_workers 2 

echo "========================================================="


echo "==================== Experiment 27 ===================="
echo "____________________Train____________________"
python test.py --test_csv ../Speech/dist/lab/train.csv -r ../Speech/dist/wav_so-denoise --checkpoint_path ../Speech/Checkpoints/Experiment-27/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-27/*/config.json  --batch_size 16 --num_workers 2 

echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav_so-denoise --checkpoint_path ../Speech/Checkpoints/Experiment-27/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-27/*/config.json  --batch_size 16 --num_workers 2 

echo "========================================================="



echo "==================== Experiment 28 ===================="
echo "____________________Train____________________"
python test.py --test_csv ../Speech/dist/lab/train.csv -r ../Speech/dist/wav_so-denoise --checkpoint_path ../Speech/Checkpoints/Experiment-28/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-28/*/config.json  --batch_size 16 --num_workers 2 

echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav_so-denoise --checkpoint_path ../Speech/Checkpoints/Experiment-28/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-28/*/config.json  --batch_size 16 --num_workers 2 

echo "========================================================="



echo "==================== Experiment 29 ===================="
echo "____________________Train____________________"
python test.py --test_csv ../Speech/dist/lab/train.csv -r ../Speech/dist/wav_so-denoise --checkpoint_path ../Speech/Checkpoints/Experiment-29/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-29/*/config.json  --batch_size 16 --num_workers 2 

echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav_so-denoise --checkpoint_path ../Speech/Checkpoints/Experiment-29/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-29/*/config.json  --batch_size 16 --num_workers 2 

echo "========================================================="

echo "==================== Experiment 30 ===================="
echo "____________________Train____________________"
python test.py --test_csv ../Speech/dist/lab/train.csv -r ../Speech/dist/wav_so-denoise --checkpoint_path ../Speech/Checkpoints/Experiment-30/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-30/*/config.json  --batch_size 16 --num_workers 2 

echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav_so-denoise --checkpoint_path ../Speech/Checkpoints/Experiment-30/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-30/*/config.json  --batch_size 16 --num_workers 2 

echo "========================================================="


echo "==================== Experiment 32 ===================="
echo "____________________Train____________________"
python test.py --test_csv ../Speech/dist/lab/train.csv -r ../Speech/dist/wav_so-denoise --checkpoint_path ../Speech/Checkpoints/Experiment-32/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-32/*/config.json  --batch_size 16 --num_workers 2 

echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav_so-denoise --checkpoint_path ../Speech/Checkpoints/Experiment-32/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-32/*/config.json  --batch_size 16 --num_workers 2 

echo "========================================================="



echo "==================== Experiment X ===================="
echo "____________________Train____________________"
python test.py --test_csv ../Speech/dist/lab/train.csv -r ../Speech/dist/wav_so-denoise --checkpoint_path ../Speech/Checkpoints/Experiment-X/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-X/*/config.json  --batch_size 16 --num_workers 2 

echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav_so-denoise --checkpoint_path ../Speech/Checkpoints/Experiment-X/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-X/*/config.json  --batch_size 16 --num_workers 2 

echo "========================================================="

# python test.py --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav_normalized --checkpoint_path ../Speech/Checkpoints/Experiment-53-all-testes/-32/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-53-all-testes/-32/*/config.json  --batch_size 16 --num_workers 2 

Experiment-53-all-testes/-32
MULTILINE-COMMENT


echo "==================== Experiment 31 ===================="
echo "____________________Train____________________"
python test.py --test_csv ../Speech/dist/lab/train.csv -r ../Speech/dist/wav_so-denoise --checkpoint_path ../Speech/Checkpoints/Experiment-31/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-31/*/config.json  --batch_size 16 --num_workers 2 

echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav_so-denoise --checkpoint_path ../Speech/Checkpoints/Experiment-31/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-31/*/config.json  --batch_size 16 --num_workers 2 

echo "========================================================="



echo "==================== Experiment 58 ===================="
echo "____________________Train____________________"
python test.py --test_csv ../Speech/dist/lab/train.csv -r ../Speech/dist/wav_normalized --checkpoint_path ../Speech/Checkpoints/Experiment-58/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-58/*/config.json  --batch_size 16 --num_workers 2 

echo "____________________EVALUATION____________________"
# without noise
python test.py --test_csv ../Speech/dist/lab/devel.csv -r ../Speech/dist/wav_normalized --checkpoint_path ../Speech/Checkpoints/Experiment-58/*/best_checkpoint.pt --config_path ../Speech/Checkpoints/Experiment-58/*/config.json  --batch_size 16 --num_workers 2 

echo "========================================================="



