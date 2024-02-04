import numpy as np
import math
import torch

def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)
        
def get_grad_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'Discrete':
        obs_shape = (obs_space.n,)
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c


def padding(x, pad_width, axis):
    axis_list = [(0, 0) for _ in range(len(x.shape))]
    axis_list[axis] = (0, pad_width)
    return np.pad(x, axis_list, 'constant', constant_values=0)


def split_into_length(x, length, axis):
    size = x.shape[axis]
    new_x = []
    for i in range(size - length + 1):
        new_x.append(np.take(x, range(i, i + length), axis=axis))
    for i in range(size - length + 1, size):
        new_x.append(padding(np.take(x, range(i, size), axis=axis), length - size + i, axis))
    return np.stack(new_x, axis=axis)


def split_n_into_length(x, length, mask=None):
    # x: [T, bsz, ...] -> [T, 2*length+1, bsz, ...]; mask: [T, bsz]
    T, N = x.shape[0], x.shape[1]
    ndim = len(x.shape)
    
    new_x = []
    for i in range(length):
        pad_width = [(0, 0) for _ in range(ndim)]
        pad_width[0] = (length - i, 0)
        x_seq = np.pad(x[:i + length + 1].copy(), pad_width, 'edge')
        if mask is not None:
            for j in range(N):
                ind = np.where(mask[:i + length + 1, j] == 0)[0]
                if len(ind) == 0:
                    continue
                ind = ind[0]
                if ind <= i:    # mask before current step
                    x_seq[:length - i + ind, j] = x_seq[length - i + ind, j]
                else:           # mask after current step
                    x_seq[length - i + ind:, j] = x_seq[length - i + ind - 1, j]
        new_x.append(x_seq)
    for i in range(length, T - length):
        x_seq = x[i - length: i + length + 1].copy()
        if mask is not None:
            for j in range(N):
                ind = np.where(mask[i - length: i + length + 1, j] == 0)[0]
                if len(ind) == 0:
                    continue
                ind = ind[0]
                if i - length + ind <= i:    # mask before current step
                    x_seq[:ind, j] = x_seq[ind, j]
                else:           # mask after current step
                    x_seq[ind:, j] = x_seq[ind - 1, j]
        new_x.append(x_seq)
    for i in range(T - length, T):
        pad_width = [(0, 0) for _ in range(ndim)]
        pad_width[0] = (0, i + length + 1 - T)
        x_seq = np.pad(x[i - length:].copy(), pad_width, 'edge')
        if mask is not None:
            for j in range(N):
                ind = np.where(mask[i - length:, j] == 0)[0]
                if len(ind) == 0:
                    continue
                ind = ind[0]
                if i - length + ind <= i:
                    x_seq[:ind, j] = x_seq[ind, j]
                else:
                    x_seq[ind:, j] = x_seq[ind - 1, j]
        new_x.append(x_seq)
    return np.stack(new_x, axis=0)


def split_o_into_length(x, length, mask=None):
    # x: [T, bsz, ...] -> [T, 2*length+1, bsz, ...]; mask: [T, bsz]
    T, N = x.shape[0], x.shape[1]
    ndim = len(x.shape)
    
    new_x = []
    for i in range(length):
        pad_width = [(0, 0) for _ in range(ndim)]
        pad_width[0] = (length - i, 0)
        x_seq = np.pad(x[:i + length + 1].copy(), pad_width, constant_values=-1)
        if mask is not None:
            for j in range(N):
                ind = np.where(mask[:i + length + 1, j] == 0)[0]
                if len(ind) == 0:
                    continue
                ind = ind[0]
                if ind <= i:    # mask before current step
                    x_seq[:length - i + ind, j] = -1
                else:           # mask after current step
                    x_seq[length - i + ind:, j] = -1
        new_x.append(x_seq)
    for i in range(length, T - length):
        x_seq = x[i - length: i + length + 1].copy()
        if mask is not None:
            for j in range(N):
                ind = np.where(mask[i - length: i + length + 1, j] == 0)[0]
                if len(ind) == 0:
                    continue
                ind = ind[0]
                if i - length + ind <= i:    # mask before current step
                    x_seq[:ind, j, :] = -1
                else:           # mask after current step
                    x_seq[ind:, j, :] = -1
        new_x.append(x_seq)
    for i in range(T - length, T):
        pad_width = [(0, 0) for _ in range(ndim)]
        pad_width[0] = (0, i + length + 1 - T)
        x_seq = np.pad(x[i - length:].copy(), pad_width, constant_values=-1)
        if mask is not None:
            for j in range(N):
                ind = np.where(mask[i - length:, j] == 0)[0]
                if len(ind) == 0:
                    continue
                ind = ind[0]
                if i - length + ind <= i:
                    x_seq[:ind, j, :] = -1
                else:
                    x_seq[ind:, j, :] = -1
        new_x.append(x_seq)
    return np.stack(new_x, axis=0)


def get_active_novel_masks_from_masks(masks, step):
    active_novel_masks = np.ones_like(masks[:-1], dtype=np.int32)
    active_novel_masks[-step+1: ] = 0
    for i in range(1, step):
        active_novel_masks[:-i][masks[i:-1] == 0] = 0
    return active_novel_masks