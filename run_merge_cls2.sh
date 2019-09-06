export CUDA_VISIBLE_DEVICES=0

python merge.py --dataset siim --seed 1234 \
                --root_results results_cls \
                --thresh 0.5 --non_empty_ratio 0.18 \
                --batch_size 1 --split test #--gt

python convert_to_only_empty.py --file_path results_cls/merged_test_5_0.5_0.18.csv
