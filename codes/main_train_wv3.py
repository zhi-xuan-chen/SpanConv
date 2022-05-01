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

import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_wv3 import Dataset_Pro
import scipy.io as sio
from model_wv3 import LightNet
import numpy as np
import shutil
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# ================== Pre-test =================== #
def summaries(model, writer=None, grad=False):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[(1,64,64),(8,64,64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    if writer is not None:
        x = torch.randn(1, 64, 64, 64)
        writer.add_graph(model, (x,))

def load_set(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8

    # tensor type:
    lms = torch.from_numpy(data['lms'] / 2047.0).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms = torch.from_numpy((data['ms'] / 2047.0)).permute(2, 0, 1)  # CxHxW= 8x64x64
    pan = torch.from_numpy((data['pan'] / 2047.0))   # HxW = 256x256

    return lms, ms, pan

def load_gt_compared(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8

    # tensor type:
    test_gt = torch.from_numpy(data['gt'] / 2047.0)  # CxHxW = 8x256x256

    return test_gt

# ================== Pre-Define =================== #
SEED = 3047
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True
weight_dir = './Weights/wv3' # directory to save trained model to.
log_dir = './train_logs/wv3' # directory to save log to.

# ============= HYPER PARAMS(Pre-Defined) ==========#
lr = 0.0025
epochs = 800
ckpt = 20 # Save the model every 'ckpt' epochs
batch_size = 8

# ============= Construct Model ==========#

model = LightNet().cuda()
#model.load_state_dict(torch.load('/Data/Machine Learning/Zhi-Xuan Chen/LAConv_ddf/DKNET.pth'), strict = False)
summaries(model, grad=True)
criterion = nn.L1Loss().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999))   # optimizer 1
scheduler = lr_scheduler.StepLR(optimizer, step_size=120, gamma=0.75)   #dynamic lr


if os.path.exists(log_dir):  # for tensorboard: copy dir of train_logs
    shutil.rmtree(log_dir)  # ---> console (see tensorboard): tensorboard --logdir = dir of train_logs
writer = SummaryWriter(log_dir)  # Tensorboard_show:


def save_checkpoint(model, epoch):  # save model function
    model_out_path = weight_dir+'/weight-'+str(epoch)+'.pth'
    if not os.path.exists(model_out_path):
        os.makedirs(model_out_path)
    torch.save(model.state_dict(), model_out_path)


###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################

def train(training_data_loader, validate_data_loader):
    print('Start training...')

    for epoch in range(epochs):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):

            gt, lms, _, _, pan = Variable(batch[0], requires_grad=False).cuda(), \
                                     Variable(batch[1]).cuda(), \
                                     batch[2], \
                                     batch[3], \
                                     Variable(batch[4]).cuda()
            optimizer.zero_grad()  # fixed
            out = model(pan, lms)

            loss = criterion(out, gt)  # compute loss
            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch

            loss.backward()  # fixed
            optimizer.step()  # fixed

        scheduler.step()  # update lr

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss

        writer.add_scalar('train/loss', t_loss, epoch)  # write to tensorboard to check
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))  # print loss for each epoch

        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, epoch)

        # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(validate_data_loader, 1):
                gt, lms, _, _ ,pan= Variable(batch[0], requires_grad=False).cuda(), \
                                         Variable(batch[1]).cuda(), \
                                         batch[2], \
                                         batch[3], \
                                         Variable(batch[4]).cuda()

                out = model(pan,lms)

                loss = criterion(out, gt)
                epoch_val_loss.append(loss.item())

        v_loss = np.nanmean(np.array(epoch_val_loss))

        writer.add_scalar('val/loss', v_loss, epoch)
        print('validate loss: {:.7f}'.format(v_loss))

    writer.close()  # close tensorboard

###################################################################training_data
# ----------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":

    train_set = Dataset_Pro('./training_data/train_wv3_10000.h5')  # creat data for training
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    validate_set = Dataset_Pro('./training_data/valid_wv3_10000.h5')  # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    train(training_data_loader, validate_data_loader)  # call train function (call: Line 97)


