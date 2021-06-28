python3 train.py --model_type satrn-large --datasets synthtext+mjsynth --resize --grayscale --gpus 4 --bs 16 --epochs 4 --save_best_model --lr 1e-5 --exp_name base_model
#python3 train.py --model_type satrn-large --datasets synthtext+mjsynth --resize --grayscale --gpus 0 --bs 16 --epochs 1 --load_weights_from './saved_models/exp-epoch=00-val_acc=0.98.ckpt' --run_test
