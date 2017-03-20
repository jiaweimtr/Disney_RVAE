## This is a project in Disney Research Pittsburgh
## by Jiawei (Eric) He
## Version 4: conditional VAE performed on the customized data set. The condition is on the previous frame.
    #AKA, first-order HMM.

import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from moving_MNIST import moving_mnist


m_mnist = moving_mnist()
print(m_mnist.shape)

mb_size = 200
Z_dim = 100
X_dim = m_mnist.shape[1]  #4096
y_dim = m_mnist.shape[1]  #4096 (y is the previous image input)
h_dim = 400
cnt = 0
lr = 1e-3



def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)


# =============================== Q(z|X) ======================================

Wxh = xavier_init(size=[X_dim + y_dim, h_dim])
bxh = Variable(torch.zeros(h_dim), requires_grad=True)

Whz_mu = xavier_init(size=[h_dim, Z_dim])
bhz_mu = Variable(torch.zeros(Z_dim), requires_grad=True)

Whz_var = xavier_init(size=[h_dim, Z_dim])
bhz_var = Variable(torch.zeros(Z_dim), requires_grad=True)


def Q(X, c):
    inputs = torch.cat([X, c], 1)
    h = nn.relu(inputs @ Wxh + bxh.repeat(inputs.size(0), 1))
    z_mu = h @ Whz_mu + bhz_mu.repeat(h.size(0), 1)
    z_var = h @ Whz_var + bhz_var.repeat(h.size(0), 1)
    return z_mu, z_var


def sample_z(mu, log_var):
    eps = Variable(torch.randn(mb_size, Z_dim))
    return mu + torch.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

Wzh = xavier_init(size=[Z_dim + y_dim, h_dim])
bzh = Variable(torch.zeros(h_dim), requires_grad=True)

Whx = xavier_init(size=[h_dim, X_dim])
bhx = Variable(torch.zeros(X_dim), requires_grad=True)


def P(z, c):
    inputs = torch.cat([z, c], 1)
    h = nn.relu(inputs @ Wzh + bzh.repeat(inputs.size(0), 1))
    X = nn.sigmoid(h @ Whx + bhx.repeat(h.size(0), 1))
    return X

def batch_generator(X, batch_ID, mb_size):
    if (batch_ID + mb_size) <= X.shape[0]:
        if batch_ID%20 ==0:
            X_batch = X[batch_ID+1:batch_ID+mb_size+1,:]*1.
            c_batch = X[batch_ID:batch_ID+mb_size,:]*1.
            batch_ID += mb_size + 1
        else:
            X_batch = X[batch_ID:batch_ID+mb_size,:]*1.
            c_batch = X[batch_ID-1:batch_ID+mb_size-1,:]*1.
            batch_ID += mb_size
    else:
        batch_ID = 1
        X_batch = X[batch_ID:mb_size+1,:]*1.
        c_batch = X[batch_ID-1:mb_size,:]*1.        
        batch_ID += mb_size

    return X_batch,c_batch, batch_ID

# =============================== TRAINING ====================================

params = [Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var,
          Wzh, bzh, Whx, bhx]

solver = optim.Adam(params, lr=lr)
batch_ID = 1


for it in range(100000):


    X,c, batch_ID = batch_generator(m_mnist, batch_ID, mb_size)
    X = Variable(torch.from_numpy(X))
    c = Variable(torch.from_numpy(c))

    # Forward
    z_mu, z_var = Q(X, c)
    z = sample_z(z_mu, z_var)
    X_sample = P(z, c)

    # Loss
    recon_loss = nn.binary_cross_entropy(X_sample, X, size_average=False) / mb_size
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
    loss = recon_loss + kl_loss

    # Backward
    loss.backward()

    # Update
    solver.step()

    # Housekeeping
    for p in params:
        p.grad.data.zero_()

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; Loss: {:.4}'.format(it, loss.data[0]))

        c = np.zeros(shape=[mb_size, y_dim], dtype='float32')

        c = m_mnist[0:mb_size,:]
        #c[:, np.random.randint(0, 10)] = 1.
        c = Variable(torch.from_numpy(c))
        z = Variable(torch.randn(mb_size, Z_dim))
        samples = P(z, c).data.numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(64, 64), cmap='Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)