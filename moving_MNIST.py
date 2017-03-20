from __future__ import division
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import fnmatch
import random

def moving_mnist():

    mnist_file = 'mnist_test_seq.npy'

    m_mnist = np.load(mnist_file)

    # shape: 20 * 10000 * 64 * 64
    len_data = m_mnist.shape[0]
    num_data = m_mnist.shape[1]
    image_h  = m_mnist.shape[2]
    image_w  = m_mnist.shape[3]

    m_mnist_reshaped = np.zeros((len_data*num_data,image_h * image_w))

    for i in range(0, num_data):
        for j in range(0, len_data):
            m_mnist_reshaped[(i)*20 + j,:] = m_mnist[j,i,:].reshape(image_h * image_w)
            #print("precessing {} out of 200,000 images".format((i)*20 + j+1))

    # m_mnist_reshaped = m_mnist.reshape(len_data*num_data,image_h * image_w)

    a = np.array(m_mnist_reshaped, dtype = np.float32)
    a = a/255.

    #plt.imshow(a[20,:].reshape(64,64))
    #plt.show()
    return a

    # print(m_mnist_reshaped.shape)
    # plt.imshow(m_mnist_reshaped[0,:,:])
    # plt.show()
    ## initialization:
    # data_list = []

    # generate a shapre ID to reshape the data
    # rand_id = np.empty((0), dtype = int)
    # for i in range(num_data):
    #     rand_id = np.append(rand_id, i)
    # random.shuffle(rand_id)

######################################

    # for i in rand_id:
    #     ID = rand_id[i]
    #     data_point = m_mnist[:,i,:,:]

    #     cont_data = np.empty((0), dtype = int)

    #     for j in range(len_data):
    #         cont_data = np.append(cont_data, 0)
    #     cont_data[-1] = 1

    #     data_point =[cont_data, data_point]
    #     data_list.append(data_point)

    # return data_list
