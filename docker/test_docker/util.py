import itertools, imageio, torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
#from scipy.misc import imresize
import math

def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r
    
    
def show_result(G, x_, y_, num_epoch, num_show=5, show = False, save = False, path = 'result.png'):
    matplotlib.use('Agg')
    #G.eval()
    test_images = G(x_)
    #print('checkpoint1')
    #num_show = x_.size()[0]
    size_figure_grid = 3
    fig, ax = plt.subplots(num_show, size_figure_grid, figsize=(5, 5))
    #print('checkpoint2')
    for i, j in itertools.product(range(num_show), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    #print('checkpoint3')
    #print(x_.shape)
    #print(test_images.shape)
    #print(y_.shape)
    for i in range(num_show):
        #print('checkpoint 3.0')
        ax[i, 0].cla()
        #ax[i, 0].imshow((x_[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        ncct = (x_[i].cpu().data.numpy())
        #print(ncct.shape)
        ax[i, 0].imshow((ncct.transpose(1, 2, 0) + 1) / 2)
        #print('checkpoint 3.1')
        ax[i, 1].cla()
        testimg = test_images[i].cpu().data.numpy().squeeze()
        #print(testimg.shape)
        #print((np.stack((testimg,)*3,0)).shape)
        ax[i, 1].imshow((np.stack((testimg,)*3,0).transpose(1,2,0) + 1) / 2)
        #print('checkpoint 3.2')
        ax[i, 2].cla()
        yimg = y_[i].numpy().squeeze()
        ax[i, 2].imshow((np.stack((yimg,)*3,0).transpose(1,2,0) + 1) / 2)
        #print('checkpoint 3.3')
        #print((np.stack((yimg,)*3,0).transpose(1,2,0).shape))

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    #print('checkpoint4')
    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    matplotlib.use('Agg')
    #print(hist)
    x = range(len(hist['G_losses']))

    y1 = hist['G_losses']
    y2 = hist['D_losses']
    
    plt.plot(x, y1, label='G_loss')
    plt.plot(x, y2, label='D_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=1)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
        
def show_train_hist_epoch(hist, show = False, save = False, path = 'Train_hist.png'):
    matplotlib.use('Agg')
    x = range(len(hist['G_loss_epoch']))

    y1 = hist['G_loss_epoch']
    y2 = hist['D_loss_epoch']
    y3 = hist['val_loss_epoch']
    y4 = hist['Extrema_loss_epoch']
    y5 = hist['Multimodal_loss_epoch']
    y6 = hist['Overall_G_loss']
    
    plt.plot(x, y1, label='G_loss')
    plt.plot(x, y2, label='D_loss')
    plt.plot(x,y4, label='extrema_loss')
    plt.plot(x,y5, label='multimodal_loss')
    plt.plot(x, y6, label='Total_G_loss')
    plt.plot(x, y3, label='val_loss')
    

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=1)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def generate_animation(root, model, opt):
    images = []
    for e in range(opt.train_epoch):
        img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(root + model + 'generate_animation.gif', images, fps=5)

def data_load(path, subfolder, transform, batch_size, shuffle=True, return_dset=False, num_workers=1):
    print(path, subfolder)
    dset = datasets.ImageFolder(path, transform)
    ind = dset.class_to_idx[subfolder]

    n = 0
    for i in range(dset.__len__()):
        if ind != dset.imgs[n][1]:
            del dset.imgs[n]
            n -= 1

        n += 1

    if not return_dset:
      return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    else:
      return dset

"""
def imgs_resize(imgs, resize_scale = 286):
    outputs = torch.FloatTensor(imgs.size()[0], imgs.size()[1], resize_scale, resize_scale)
    for i in range(imgs.size()[0]):
        img = imresize(imgs[i].numpy(), [resize_scale, resize_scale])
        outputs[i] = torch.FloatTensor((img.transpose(2, 0, 1).astype(np.float32).reshape(-1, imgs.size()[1], resize_scale, resize_scale) - 127.5) / 127.5)

    return outputs
"""

def random_crop(imgs1, imgs2, crop_size = 256):
    outputs1 = torch.FloatTensor(imgs1.size()[0], imgs1.size()[1], crop_size, crop_size)
    outputs2 = torch.FloatTensor(imgs2.size()[0], imgs2.size()[1], crop_size, crop_size)
    for i in range(imgs1.size()[0]):
        img1 = imgs1[i]
        img2 = imgs2[i]
        rand1 = np.random.randint(0, imgs1.size()[2] - crop_size)
        rand2 = np.random.randint(0, imgs2.size()[2] - crop_size)
        outputs1[i] = img1[:, rand1: crop_size + rand1, rand2: crop_size + rand2]
        outputs2[i] = img2[:, rand1: crop_size + rand1, rand2: crop_size + rand2]

    return outputs1, outputs2

def random_fliplr(imgs1, imgs2):
    outputs1 = torch.FloatTensor(imgs1.size())
    outputs2 = torch.FloatTensor(imgs2.size())
    for i in range(imgs1.size()[0]):
        if torch.rand(1)[0] < 0.5:
            img1 = torch.FloatTensor(
                (np.fliplr(imgs1[i].numpy().transpose(1, 2, 0)).transpose(2, 0, 1).reshape(-1, imgs1.size()[1], imgs1.size()[2], imgs1.size()[3]) + 1) / 2)
            outputs1[i] = (img1 - 0.5) / 0.5
            img2 = torch.FloatTensor(
                (np.fliplr(imgs2[i].numpy().transpose(1, 2, 0)).transpose(2, 0, 1).reshape(-1, imgs2.size()[1], imgs2.size()[2], imgs2.size()[3]) + 1) / 2)
            outputs2[i] = (img2 - 0.5) / 0.5
        else:
            outputs1[i] = imgs1[i]
            outputs2[i] = imgs2[i]

    return outputs1, outputs2
