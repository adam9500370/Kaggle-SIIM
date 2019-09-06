import pandas as pd
import os
import argparse
import collections


parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--file_path_empty', nargs='?', type=str, default='',
                    help='CSV file path for empty masks')
parser.add_argument('--file_path_non_empty', nargs='?', type=str, default='',
                    help='CSV file path for non-empty masks')
args = parser.parse_args()
print(args)


root_path = os.path.join('results_combined')
if not os.path.exists(root_path):
    os.makedirs(root_path)

##leak_df = pd.read_csv(os.path.join('..', 'datasets', 'siim', 'sample_submission_leak.csv'), index_col=0)
empty_df = pd.read_csv(args.file_path_empty, index_col=0)
non_empty_df = pd.read_csv(args.file_path_non_empty, index_col=0)
empty_dict = empty_df.to_dict('index')
non_empty_dict = non_empty_df.to_dict('index')

new_dict = collections.OrderedDict()
num_empty = 0
missed_non_empty = 0
num_non_empty = 0
##num_leak = 0
for d in empty_dict:
    """
    if len(leak_df[leak_df.index == d]['EncodedPixels']) > 1:
        num_leak += 1
        non_empty_mask = non_empty_dict[d]['EncodedPixels']
        if '0 1' in non_empty_mask[:3]:
            missed_non_empty += 1
            new_dict[d] = '-1'
        else:
            new_dict[d] = non_empty_mask
            num_non_empty += 1
        continue
    #"""

    if '-1' in empty_dict[d]['EncodedPixels']:
        new_dict[d] = '-1'
        num_empty += 1
    else:
        non_empty_mask = non_empty_dict[d]['EncodedPixels']
        if '0 1' in non_empty_mask[:3]:
            missed_non_empty += 1
            new_dict[d] = '-1'
        else:
            new_dict[d] = non_empty_mask
            num_non_empty += 1

print('empty: {}; non-empty: {}; missed non-empty: {}'.format(num_empty, num_non_empty, missed_non_empty))
##print('empty: {}; non-empty: {}; missed non-empty: {}; leak: {}'.format(num_empty, num_non_empty, missed_non_empty, num_leak))

sub = pd.DataFrame.from_dict(new_dict, orient='index')
sub.index.names = ['ImageId']
sub.columns = ['EncodedPixels']
sub.to_csv(os.path.join(root_path, 'combined.csv'))
##sub.to_csv(os.path.join(root_path, 'combined+leak.csv'))
