## This is a project in Disney Research Pittsburgh
## by Jiawei (Eric) He
## Version 3: Vanilla VAE with customized input (the input is each frame in moving-MNIST dataset)




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
#from tensorflow.examples.tutorials.mnist import input_data

m_mnist = moving_mnist()

mb_size = 200
Z_dim = 100
X_dim = m_mnist.shape[1]  #4096
h_dim = 128
c = 0
lr = 1e-3


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)


# =============================== Q(z|X) ======================================

Wxh = xavier_init(size=[X_dim, h_dim])                           # 4096 * 400
bxh = Variable(torch.zeros(h_dim), requires_grad=True)           # 400          

Whz_mu = xavier_init(size=[h_dim, Z_dim])                        # 400 * 20                      
bhz_mu = Variable(torch.zeros(Z_dim), requires_grad=True)        # 20

Whz_var = xavier_init(size=[h_dim, Z_dim])						 # 400 * 20
bhz_var = Variable(torch.zeros(Z_dim), requires_grad=True)		 # 20


def Q(X):

    h = nn.relu(X @ Wxh + bxh.repeat(X.size(0), 1))
    z_mu = h @ Whz_mu + bhz_mu.repeat(h.size(0), 1)
    z_var = h @ Whz_var + bhz_var.repeat(h.size(0), 1)
    return z_mu, z_var


def sample_z(mu, log_var):
    eps = Variable(torch.randn(mb_size, Z_dim))
    return mu + torch.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

Wzh = xavier_init(size=[Z_dim, h_dim])
bzh = Variable(torch.zeros(h_dim), requires_grad=True)

Whx = xavier_init(size=[h_dim, X_dim])
bhx = Variable(torch.zeros(X_dim), requires_grad=True)


def P(z):
    h = nn.relu(z @ Wzh + bzh.repeat(z.size(0), 1))
    X = nn.sigmoid(h @ Whx + bhx.repeat(h.size(0), 1))
    return X


# =============================== Batch Generator===============================


def batch_generator(X, batch_ID, mb_size):
    if (batch_ID + mb_size) <= X.shape[0]:
        X_batch = X[batch_ID:batch_ID+mb_size,:]
        batch_ID += mb_size
    else:
        batch_ID = 0
        X_batch = X[0:mb_size,:]*1.
        batch_ID += mb_size

    return X_batch, batch_ID


# =============================== TRAINING ====================================

params = [Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var,
          Wzh, bzh, Whx, bhx]

# solver = optim.Adam(params, lr=lr)

batch_ID = 0

for it in range(100000):

    learning_rate =lr * (0.1**(int(it/10000)))
    solver = optim.Adam(params, learning_rate)



    X, batch_ID = batch_generator(m_mnist, batch_ID, mb_size)
    X = Variable(torch.from_numpy(X))

	#new_X = X * Wxh + bxh.repeat(X.size(0), 1)
    # Forward
    z_mu, z_var = Q(X)

    z = sample_z(z_mu, z_var)

    X_sample = P(z)

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
        print("learning rate is {}".format(learning_rate))

        samples = P(z).data.numpy()[:16]

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

        plt.savefig('out/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
        c += 1
        plt.close(fig)