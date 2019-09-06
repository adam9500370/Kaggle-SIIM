export CUDA_VISIBLE_DEVICES=0

#i=0
k=5
img_size=768
model_name=bisenet-resnet18
norm_type=bn
saved_iter=best-wacc
thresh=0.5
non_empty_ratio=0.17
root_ckpts=checkpoints_cls
root_results=results_cls

for((i=0;i<${k};i=i+1))
do
#python test.py --model_path ${root_ckpts}/${model_name}_siim_${saved_iter}_${img_size}x${img_size}_${i}-${k}_model.pth --norm_type ${norm_type} --dataset siim \
#               --img_rows ${img_size} --img_cols ${img_size} --seed 1234 --fold_num $i --num_folds $k \
#               --root_results ${root_results} \
#               --thresh ${thresh} --non_empty_ratio ${non_empty_ratio} --batch_size 1 --split val --gt #--only_non_empty #--use_cbam #--tta

python test.py --model_path ${root_ckpts}/${model_name}_siim_${saved_iter}_${img_size}x${img_size}_${i}-${k}_model.pth --norm_type ${norm_type} --dataset siim \
               --img_rows ${img_size} --img_cols ${img_size} --seed 1234 --fold_num $i --num_folds $k \
               --root_results ${root_results} \
               --thresh ${thresh} --non_empty_ratio ${non_empty_ratio} --batch_size 1 --split test #--only_non_empty #--use_cbam #--tta #--gt
done
