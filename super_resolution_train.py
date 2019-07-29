
# coding: utf-8

# # image super resolution

import matplotlib.pylab as plt
import seaborn as sns
from helper import *
from unet import *
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from skimage import io
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0)

args = parser.parse_args()

# %matplotlib inline
# plt.rcParams['figure.figsize'] = 10, 8

class ProbavData(Dataset):

    def __init__(self, folder_dataset, train=None, transform=None):
        self.transform = transform
        self.train = train

        if self.train:
            self.__xs = []
            self.__ys = []
            train = all_scenes_paths(folder_dataset + 'train')
            for i in range(len(train)):
                path = train[i]
                path = path if path[-1] in {'/', '\\'} else (path + '/')
                for f in glob(path + 'LR*.png'):
                    q = f.replace('LR', 'QM')
                    lr = np.stack((f, q))
                    self.__xs.append(lr)
                    hr = np.stack((path + 'HR.png', path + 'SM.png'))
                    self.__ys.append(hr)

        else:
            self.__xs_test = []
            self.__ymask_test = []
            test  = all_scenes_paths(folder_dataset + 'test')
            for i in range(len(test)):
                path = test[i]
                self.__xs_test.append(path)


    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):

        if self.train:
            lr = skimage.io.imread(self.__xs[index][0])
            lr_mask = skimage.io.imread(self.__xs[index][1], dtype=np.bool)
            hr = skimage.io.imread(self.__ys[index][0])
            hr_mask = skimage.io.imread(self.__ys[index][1], dtype=np.bool)

            hr = skimage.img_as_float(hr)
            hr = hr.astype(np.float32)
            hr = hr.reshape(1, 384, 384)
            hr = torch.from_numpy(np.asarray(hr))
            hr = hr.type(torch.FloatTensor)

            lr = skimage.img_as_float(lr)
            lr = lr.astype(np.float32)
            lr = lr.repeat(3, axis = 0).repeat(3, axis = 1)

            lr_mask = lr_mask*1
            lr_mask = lr_mask.repeat(3, axis = 0).repeat(3, axis = 1)

            lr_reshape = lr.reshape(1, 384, 384)
            lr_mask = lr_mask.reshape(1, 384, 384)
            image = np.vstack((lr_reshape, lr_mask))

            hr_mask = hr_mask*1
            hr_mask = hr_mask.reshape(1, 384, 384)

            if self.transform is not None:
                img = self.transform(img)

            # Convert image and label to torch tensors
            image = torch.from_numpy(np.asarray(image))
            image = image.type(torch.FloatTensor)

            hr_mask = torch.from_numpy(np.asarray(hr_mask))
            hr_mask = hr_mask.type(torch.FloatTensor)

        else:
            #lr = skimage.io.imread(self.__xs_test[index])
            lr = central_tendency(self.__xs_test[index], agg_with='median', only_clear=True)
            lr = lr.astype(np.float32)
            lr = lr.repeat(3, axis = 0).repeat(3, axis = 1)
            lr_reshape = lr.reshape(1, 384, 384)
            # write a mask filled with 1.
            lr_mask = np.ones(147456).reshape(1, 384, 384)
            image = np.vstack((lr_reshape, lr_mask))
             # Convert image and label to torch tensors
            image = torch.from_numpy(np.asarray(image))
            image = image.type(torch.FloatTensor)

        if self.train:
            return image, hr, hr_mask

        else:
            return image

    # Override to give PyTorch size of dataset
    def __len__(self):
        if self.train:
            return len(self.__xs)
        else:
            return len(self.__xs_test)

train_dataset = ProbavData(folder_dataset = 'probav_data/', train=True)
test_dataset = ProbavData(folder_dataset = 'probav_data/', train=False)

data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=args.batch_size,
                                          num_workers=0,
                                          drop_last=True,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          num_workers=0,
                                          drop_last=True,
                                          shuffle=False)

# In[4]:
def build_model():
    net = UNet(2, 1)
    if torch.cuda.is_available():
        net.cuda()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=args.weight_decay, amsgrad=False)
    return net, opt

def weighted_mse_loss(input,target,weights):
    out = (input-target)**2
    out = out * weights
    loss = out.mean()
    return loss

## Build the model
net, opt = build_model()

epoch_train_loss = []

for epoch in tqdm(range(args.num_epoch)):
    print('Starting epoch {}/{}.'.format(epoch + 1, args.num_epoch))

    net.train()
    epoch_loss = 0

    for i, (image, label, label_mask) in enumerate(data_loader):

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            label_mask = label_mask.cuda()

        label_mask = label_mask.view(-1)

        pred = net(image)
        probs_flat = pred.view(-1)
        true_flat = label.view(-1)
        loss = weighted_mse_loss(probs_flat, true_flat, label_mask)
        epoch_loss += loss.item()

        opt.zero_grad()
        loss.backward()
        opt.step()

    print('Epoch finished ! Loss: {}'.format(epoch_loss / i))
    epoch_train_loss.append(loss.item())
    epoch_train_loss_arr = np.array(epoch_train_loss)

def build_file_name(lr, batch_size, weight_decay, num_epoch, model):

    if model:
        format = '.pt'
    else:
        format = '.txt'

    return (os.path.dirname(os.path.realpath(__file__)) +
            '/result/' + '/' + 'lr' + str(lr) + 'bs' + str(batch_size) + 'wd' + str(weight_decay) + 'num_epoch' + str(num_epoch) + format)

txt_dir = build_file_name(args.lr, args.batch_size, args.weight_decay, args.num_epoch, model=False)
np.savetxt(txt_dir, epoch_train_loss_arr, fmt='%s')

model_dir = build_file_name(args.lr, args.batch_size, args.weight_decay, args.num_epoch, model=True)
#Save the model
torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': opt.state_dict()
            }, model_dir)
