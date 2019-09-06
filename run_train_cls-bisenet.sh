export CUDA_VISIBLE_DEVICES=0

k=5
for((i=0;i<${k};i=i+1))
do 
python train.py --arch bisenet-resnet18 --norm_type bn --dataset siim \
                --img_rows 768 --img_cols 768 --in_channels 3 \
                --n_iter 111200 --batch_size 6 --seed 1234 \
                --l_rate 5e-4 --weight_decay 1e-4 --iter_size 8 \
                --num_cycles 1 --print_train_freq 1000 --eval_freq 1112 --save_freq 0.01 \
                --fold_num $i --num_folds $k --start_iter 0 \
                --alpha 0.25 --gamma 0.0 --lambda_fl 1.0 --lambda_dc 0.0 --lambda_lv 0.1 --lambda_cls 0.1 --ratio 0.5 --mask_dilation_size 128 --weight_acc_non_empty 1.0
done

mv checkpoints checkpoints_cls
