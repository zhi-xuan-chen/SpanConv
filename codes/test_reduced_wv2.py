# ---------------------------------------------------------------
# Copyright (c) 2022, Zhi-Xuan Chen, Cheng Jin, Xiao Wu, Liang-Jian Deng
# All rights reserved.
#
# This work is licensed under GNU Affero General Public License
# v3.0 International To view a copy of this license, see the
# LICENSE file.
#
# This file is running on WorldView-3 dataset. For other dataset
# (i.e., QuickBird), please change the corresponding
# inputs.
# ---------------------------------------------------------------

import torch
import numpy as np
from model_wv3 import LightNet
import h5py
import scipy.io as sio
import os

###################################################################
# ------------------- Sub-Functions (will be used) -------------------
###################################################################

def load_set(file_path):
    data = h5py.File(file_path)  # NxCxHxW = 0x1x2x3

    # tensor type:
    lms1 = data['lms'][...]  # NxCxHxW = 4x8x512x512
    print(lms1.shape)
    lms1 = np.array(lms1, dtype=np.float32) / 2047.0
    lms = torch.from_numpy(lms1)  # NxCxHxW  or HxWxC
    print(lms.shape)

    pan1 = data['pan'][...]   # NxCxHxW = 4x8x512x512
    pan1 = np.array(pan1, dtype=np.float32) / 2047.0
    pan = torch.from_numpy(pan1)
    print(pan.shape)

    gt = data['gt'][...]  # NxCxHxW = 4x8x512x512
    print(gt.shape)
    gt = np.array(gt, dtype=np.float32)
    gt = torch.from_numpy(gt)  # NxCxHxW  or HxWxC
    print(gt.shape)

    return lms, pan, gt


# ==============  Main test  ================== #
ckpt = "./Weights/wv3/weight-720.pth"   # choose a model

def test(file_path):
    lms, pan, test_gt = load_set(file_path)

    model = LightNet().eval()
    weight = torch.load(ckpt)
    model.load_state_dict(weight, strict=False)

    with torch.no_grad():

        x1, x2 = lms, pan   # read data: CxHxW (numpy type)
        x1 = x1.float()
        x2 = x2.float()

        sr = model(x2, x1)  # tensor type: sr = NxCxHxW

        # convert to numpy type with permute and squeeze: HxWxC (go to cpu for easy saving)
        sr = torch.squeeze(sr).permute(0, 2, 3, 1).cpu().detach().numpy()  # to: NxHxWxC
        sr = np.clip(sr, 0, 1)

        num_exm = sr.shape[0]

        for index in range(num_exm):  # save the LightNet results for matlab evaluate code
            file_name = "LightNet_reduced_wv2_" + str(index) + ".mat"
            directory_name = "./Results/12_reduced_wv2_512x512"

            save_name = os.path.join(directory_name, file_name)
            sio.savemat(save_name, {'LightNet_reduced_wv2_' + str(index): sr[index, :, :, :]})

###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == '__main__':

    file_path = "./test_data/12_reduced_wv2_512x512.h5"
    test(file_path)
