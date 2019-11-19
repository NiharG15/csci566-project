import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
# from tensorflow.contrib.slim import fully_connected as fc
# from tensorflow.contrib.slim import conv2d

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt 


class VariationalAutoencoder(object):

    def __init__(self, learning_rate=1e-4, batch_size=64, n_z=256):
        # Set hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z
    
    def encode2(self, image_input):
        x = tf.layers.conv2d(image_input, filters=64, kernel_size=3, strides=3, padding='same', activation=tf.nn.relu)

        x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=2, padding='same', activation='relu')
        # x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
        
        x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        # x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)

        x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        # x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)

        x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        # x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        # x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        # print('Shape: ', x.shape)
        x = tf.reduce_max(x, axis=[1, 2])
        x = tf.layers.flatten(x)

        mean = tf.layers.dense(x, units=self.n_z)
        stddev = tf.layers.dense(x, units=self.n_z, activation=tf.nn.softplus)

        return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=stddev)


    def decode2(self, z):
        with tf.name_scope('image_decoder'):
            x = tf.layers.dense(z, units=300, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=300, activation=tf.nn.relu)
            x = tf.reshape(x, [-1, 15, 20, 1])

            x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
            # x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
            # x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)

            x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)

            x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)

            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)

            x = tf.layers.conv2d_transpose(x, filters=3, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)

        return x

    def loss(self, imgs):
        dist = self.encode2(imgs)
        z = dist.sample()
        recon_imgs = self.decode2(z)
        n_images = imgs.shape[0]
        N_0_1 = tfp.distributions.MultivariateNormalDiag(loc=[0.] * self.n_z, scale_diag=[1.] * self.n_z)
        recon_loss = tf.nn.l2_loss(imgs - recon_imgs)
        kl_div = tf.reduce_mean(tf.maximum(tfp.distributions.kl_divergence(dist, N_0_1), 0))
        tf.summary.scalar('image_recon_loss', recon_loss)
        tf.summary.scalar('image_kl_div', kl_div)
        return recon_loss, kl_div
