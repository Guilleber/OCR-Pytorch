#/usr/bin/python3 train.py --model_type satrn-large-resnet --datasets full_words --resize --width -1 --gpus 4 --bs 16 --epochs 2 --lr 5e-6 --exp_name hc_full_cl --num_workers 4 --save_best_model --length_limit 2
#/usr/bin/python3 train.py --model_type satrn-large-resnet --datasets full_words --resize --width -1 --gpus 4 --bs 16 --epochs 4 --lr 5e-6 --exp_name hc_full_cl --num_workers 4 --save_best_model --length_limit 5 --resume_from ./saved_models/hc_full_cl.ckpt
#/usr/bin/python3 train.py --model_type satrn-large-resnet --datasets full_words --resize --width -1 --gpus 4 --bs 16 --epochs 6 --lr 5e-6 --exp_name hc_full_cl --num_workers 4 --save_best_model --length_limit 10 --resume_from ./saved_models/hc_full_cl-v1.ckpt
/usr/bin/python3 train.py --model_type satrn-large-resnet --datasets full_words --resize --width -1 --gpus 4 --bs 16 --epochs 10 --lr 5e-6 --exp_name full_words_no_augment --num_workers 4 --save_best_model --augmentation simple --load_weights_from ./saved_models/synthtext+mjsynth_2M.ckpt --run_val
#python3 train.py --model_type satrn-large --datasets synthtext+mjsynth --resize --grayscale --gpus 0 --bs 16 --epochs 1 --load_weights_from './saved_models/exp-epoch=00-val_acc=0.98.ckpt' --run_test
