export CUDA_VISIBLE_DEVICES=0

python merge.py --dataset siim --seed 1234 \
                --root_results results_seg \
                --thresh 0.5 --non_empty_ratio 1.0 \
                --batch_size 1 --split test --only_non_empty #--gt

python combine_results.py --file_path_empty results_cls/merged_test_5_0.5_0.18_only_empty.csv \
                          --file_path_non_empty results_seg/merged_test_20_0.5_1.0.csv
