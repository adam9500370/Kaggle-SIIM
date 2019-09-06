"""
Convert pretrained R-50-GN to pytorch model from https://github.com/facebookresearch/Detectron/tree/master/projects/GN
"""

import torch
import pickle
import os

with open(os.path.join('pretrained', 'R-50-GN.pkl'), 'rb') as f:
    ckpt = pickle.load(f, encoding='latin1')

##print(list(sorted(ckpt['blobs'].keys())))
"""
['conv1_gn_b', 'conv1_gn_s', 'conv1_w', 'pred_b', 'pred_w', 
'res2_0_branch1_gn_b', 'res2_0_branch1_gn_s', 'res2_0_branch1_w', 
'res2_0_branch2a_gn_b', 'res2_0_branch2a_gn_s', 'res2_0_branch2a_w', 
'res2_0_branch2b_gn_b', 'res2_0_branch2b_gn_s', 'res2_0_branch2b_w', 
'res2_0_branch2c_gn_b', 'res2_0_branch2c_gn_s', 'res2_0_branch2c_w', 
'res2_1_branch2a_gn_b', 'res2_1_branch2a_gn_s', 'res2_1_branch2a_w', 
'res2_1_branch2b_gn_b', 'res2_1_branch2b_gn_s', 'res2_1_branch2b_w', 
'res2_1_branch2c_gn_b', 'res2_1_branch2c_gn_s', 'res2_1_branch2c_w', 
'res2_2_branch2a_gn_b', 'res2_2_branch2a_gn_s', 'res2_2_branch2a_w', 
'res2_2_branch2b_gn_b', 'res2_2_branch2b_gn_s', 'res2_2_branch2b_w', 
'res2_2_branch2c_gn_b', 'res2_2_branch2c_gn_s', 'res2_2_branch2c_w', 
'res3_0_branch1_gn_b', 'res3_0_branch1_gn_s', 'res3_0_branch1_w', 
'res3_0_branch2a_gn_b', 'res3_0_branch2a_gn_s', 'res3_0_branch2a_w', 
'res3_0_branch2b_gn_b', 'res3_0_branch2b_gn_s', 'res3_0_branch2b_w', 
'res3_0_branch2c_gn_b', 'res3_0_branch2c_gn_s', 'res3_0_branch2c_w', 
'res3_1_branch2a_gn_b', 'res3_1_branch2a_gn_s', 'res3_1_branch2a_w', 
'res3_1_branch2b_gn_b', 'res3_1_branch2b_gn_s', 'res3_1_branch2b_w', 
'res3_1_branch2c_gn_b', 'res3_1_branch2c_gn_s', 'res3_1_branch2c_w', 
'res3_2_branch2a_gn_b', 'res3_2_branch2a_gn_s', 'res3_2_branch2a_w', 
'res3_2_branch2b_gn_b', 'res3_2_branch2b_gn_s', 'res3_2_branch2b_w', 
'res3_2_branch2c_gn_b', 'res3_2_branch2c_gn_s', 'res3_2_branch2c_w', 
'res3_3_branch2a_gn_b', 'res3_3_branch2a_gn_s', 'res3_3_branch2a_w', 
'res3_3_branch2b_gn_b', 'res3_3_branch2b_gn_s', 'res3_3_branch2b_w', 
'res3_3_branch2c_gn_b', 'res3_3_branch2c_gn_s', 'res3_3_branch2c_w', 
'res4_0_branch1_gn_b', 'res4_0_branch1_gn_s', 'res4_0_branch1_w', 
'res4_0_branch2a_gn_b', 'res4_0_branch2a_gn_s', 'res4_0_branch2a_w', 
'res4_0_branch2b_gn_b', 'res4_0_branch2b_gn_s', 'res4_0_branch2b_w', 
'res4_0_branch2c_gn_b', 'res4_0_branch2c_gn_s', 'res4_0_branch2c_w', 
'res4_1_branch2a_gn_b', 'res4_1_branch2a_gn_s', 'res4_1_branch2a_w', 
'res4_1_branch2b_gn_b', 'res4_1_branch2b_gn_s', 'res4_1_branch2b_w', 
'res4_1_branch2c_gn_b', 'res4_1_branch2c_gn_s', 'res4_1_branch2c_w', 
'res4_2_branch2a_gn_b', 'res4_2_branch2a_gn_s', 'res4_2_branch2a_w', 
'res4_2_branch2b_gn_b', 'res4_2_branch2b_gn_s', 'res4_2_branch2b_w', 
'res4_2_branch2c_gn_b', 'res4_2_branch2c_gn_s', 'res4_2_branch2c_w', 
'res4_3_branch2a_gn_b', 'res4_3_branch2a_gn_s', 'res4_3_branch2a_w', 
'res4_3_branch2b_gn_b', 'res4_3_branch2b_gn_s', 'res4_3_branch2b_w', 
'res4_3_branch2c_gn_b', 'res4_3_branch2c_gn_s', 'res4_3_branch2c_w', 
'res4_4_branch2a_gn_b', 'res4_4_branch2a_gn_s', 'res4_4_branch2a_w', 
'res4_4_branch2b_gn_b', 'res4_4_branch2b_gn_s', 'res4_4_branch2b_w', 
'res4_4_branch2c_gn_b', 'res4_4_branch2c_gn_s', 'res4_4_branch2c_w', 
'res4_5_branch2a_gn_b', 'res4_5_branch2a_gn_s', 'res4_5_branch2a_w', 
'res4_5_branch2b_gn_b', 'res4_5_branch2b_gn_s', 'res4_5_branch2b_w', 
'res4_5_branch2c_gn_b', 'res4_5_branch2c_gn_s', 'res4_5_branch2c_w', 
'res5_0_branch1_gn_b', 'res5_0_branch1_gn_s', 'res5_0_branch1_w', 
'res5_0_branch2a_gn_b', 'res5_0_branch2a_gn_s', 'res5_0_branch2a_w', 
'res5_0_branch2b_gn_b', 'res5_0_branch2b_gn_s', 'res5_0_branch2b_w', 
'res5_0_branch2c_gn_b', 'res5_0_branch2c_gn_s', 'res5_0_branch2c_w', 
'res5_1_branch2a_gn_b', 'res5_1_branch2a_gn_s', 'res5_1_branch2a_w', 
'res5_1_branch2b_gn_b', 'res5_1_branch2b_gn_s', 'res5_1_branch2b_w', 
'res5_1_branch2c_gn_b', 'res5_1_branch2c_gn_s', 'res5_1_branch2c_w', 
'res5_2_branch2a_gn_b', 'res5_2_branch2a_gn_s', 'res5_2_branch2a_w', 
'res5_2_branch2b_gn_b', 'res5_2_branch2b_gn_s', 'res5_2_branch2b_w', 
'res5_2_branch2c_gn_b', 'res5_2_branch2c_gn_s', 'res5_2_branch2c_w']
"""

#"""
new_ckpt = {}
new_ckpt['model_state'] = {}
for k in list(sorted(ckpt['blobs'].keys())):
    new_k = k
    if '_b' in k[-2:]:
        new_k = new_k[:-2] + '.bias'
    elif '_w' in k[-2:] or '_s' in k[-2:]:
        new_k = new_k[:-2] + '.weight'

    if 'pred' in k:
        new_k = new_k.replace('pred', 'fc')
    elif 'conv1_gn' in k:
        new_k = new_k.replace('conv1_gn', 'bn1')
    elif 'res' in k:
        new_k = new_k.replace('_', '.')
        layer, block, branch = new_k.split('.')[:3]
        new_k = new_k.replace(layer, 'layer{}'.format(int(layer[3])-1))
        if 'branch1' in branch:
            if 'gn' in k:
                new_k = new_k.replace('branch1.gn', 'downsample.1')
            else: # conv
                new_k = new_k.replace('branch1', 'downsample.0')
        else:
            if 'gn' in k:
                new_k = new_k.replace('branch2a.gn', 'bn1').replace('branch2b.gn', 'bn2').replace('branch2c.gn', 'bn3')
            else: # conv
                new_k = new_k.replace('branch2a', 'conv1').replace('branch2b', 'conv2').replace('branch2c', 'conv3')

    new_ckpt['model_state'][new_k] = torch.from_numpy(ckpt['blobs'][k])

torch.save(new_ckpt, os.path.join('pretrained', 'R-50-GN.pth'))
#"""
