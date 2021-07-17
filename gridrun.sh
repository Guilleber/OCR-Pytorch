python3 train.py --model_type satrn-large --datasets iam --resize --grayscale --width -1 --gpus 1 --bs 16 --epochs 200 --save_best_model --lr 1e-5 --exp_name iam_words
#python3 train.py --model_type satrn-large --datasets synthtext+mjsynth --resize --grayscale --gpus 0 --bs 16 --epochs 1 --load_weights_from './saved_models/exp-epoch=00-val_acc=0.98.ckpt' --run_test
