import pandas as pd
import os
import argparse
import collections


parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--file_path', nargs='?', type=str, default='results_cls/merged_test_10_0.5_0.17.csv',
                    help='CSV file path for converting to only empty masks')
args = parser.parse_args()
print(args)


root_path = args.file_path.replace(os.path.basename(args.file_path), '')
csv_name = '.'.join(os.path.basename(args.file_path).split('.')[:-1])
mix_df = pd.read_csv(args.file_path, index_col=0)
mix_dict = mix_df.to_dict('index')

new_dict = collections.OrderedDict()
for d in mix_dict:
    if '-1' in mix_dict[d]['EncodedPixels']:
        new_dict[d] = '-1'
    else:
        new_dict[d] = '0 1'

sub = pd.DataFrame.from_dict(new_dict, orient='index')
sub.index.names = ['ImageId']
sub.columns = ['EncodedPixels']
sub.to_csv(os.path.join(root_path, csv_name + '_only_empty.csv'))
