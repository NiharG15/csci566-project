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
        x = tf.layers.conv2d(image_input, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
#         print(x.shape)
        x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
        # x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
#         print(x.shape)
        x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        # x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
#         print(x.shape)
        x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        # x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
#         print(x.shape)
        x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        # print('Shape: ', x.shape)
#         print(x.shape)
        x = tf.reduce_max(x, axis=[1, 2])
#         print(x.shape)
        x = tf.layers.flatten(x)

        mean = tf.layers.dense(x, units=self.n_z)
        stddev = tf.layers.dense(x, units=self.n_z, activation=tf.nn.softplus)

        return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=stddev)


    def decode2(self, z):
        with tf.name_scope('image_decoder'):
            x = tf.layers.dense(z, units=300, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=300, activation=tf.nn.relu)
            x = tf.reshape(x, [-1, 15, 20, 1])
#             print(x.shape)
            x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
            # x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
            # x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
#             print(x.shape)
            x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
            # x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
#             print(x.shape)
            x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
            # x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
#             print(x.shape)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
            # x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
#             print(x.shape)
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
        tf.summary.scalar('total_loss', recon_loss + kl_div)
        tf.summary.image('Input Image', imgs, max_outputs=2)
        tf.summary.image('Generated Image', recon_imgs, max_outputs=2)
        return recon_loss, kl_div

      
class Autoencoder(object):

    def __init__(self, learning_rate=1e-4, batch_size=64, n_z=256):
        # Set hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z
    
    def encode2(self, image_input):
        x = tf.layers.conv2d(image_input, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)

        x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.layers.batch_normalization(x, training=True)
        # x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
        # print(x.shape)
        x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.layers.batch_normalization(x, training=True)
        # x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
        # print(x.shape)
        x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.layers.batch_normalization(x, training=True)
        # x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
        # print(x.shape)
        x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, filters=512, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.layers.batch_normalization(x, training=True)

#         print('Shape: ', x.shape)
        # print(x.shape)
        # x = tf.reduce_max(x, axis=[1, 2])
        # print(x.shape)
        print(x.shape)
        x = tf.layers.flatten(x)
        print(x.shape)
        # x = tf.layers.batch_normalization(x, training=True)
#         x = tf.layers.dense(x, units=self.n_z)
#         x = tf.layers.batch_normalization(x, training=True)
        # encoded = tf.layers.dense(x, units=self.n_z)
        return x


    def decode2(self, z):
        with tf.name_scope('image_decoder'):
            # x = tf.layers.dense(z, units=16, activation=tf.nn.relu)
           #  x = tf.layers.dense(x, units=64, activation=tf.nn.relu)
            x = tf.reshape(z, [-1, 2, 2, 512])
#             print(x.shape)
            x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
#             print(x.shape)
            x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
#             print(x.shape)
            x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
#             print(x.shape)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
#             print(x.shape)
            x = tf.layers.conv2d_transpose(x, filters=3, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)

        return x

    def loss(self, imgs):
        # encoded = self.encode2(imgs)
        # recon_imgs = self.decode2(encoded)
        # recon_imgs = self.autoencode(imgs)
        encoded, s1, s2, s3, s4 = self.encode_new(imgs)
        recon_imgs = self.decode_new(encoded, s1, s2, s3, s4)
        n_images = imgs.shape[0]
        # N_0_1 = tfp.distributions.MultivariateNormalDiag(loc=[0.] * self.n_z, scale_diag=[1.] * self.n_z)
        # econ_loss = tf.reduce_mean(tf.pow((imgs - recon_imgs), 2))
        tf.summary.image('Input Image', imgs, max_outputs=4)
        tf.summary.image('Generated Image', recon_imgs, max_outputs=4)
        flat_imgs = tf.reshape(imgs, (tf.shape(imgs)[0], -1))
        flat_recon_imgs = tf.reshape(recon_imgs, (tf.shape(recon_imgs)[0], -1))
        recon_loss = tf.reduce_sum(tf.pow(flat_imgs - flat_recon_imgs, 2), axis=1)
        recon_loss = tf.reduce_mean(recon_loss)
        tf.summary.scalar('image_recon_loss', recon_loss)

        return recon_loss
      
    def autoencode(self, imgs):
        """ This method is for experimenting with architectures """
        x = tf.layers.conv2d(imgs, filters=32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        shape2 = x.shape
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
        x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        shape1 = x.shape
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
        # shape0 = x.shape
        # x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        # x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
#         shape_orig = x.shape
#         x = tf.layers.flatten(x)
#         shape_dense = x.shape
#         x = tf.layers.dense(x, units=512, activation=tf.nn.relu)
#         x = tf.layers.dense(x, units=shape_dense[-1], activation=tf.nn.relu)
#         x = tf.reshape(x, [-1, shape_orig[1], shape_orig[2], shape_orig[3]])
        # x = tf.layers.conv2d(x, filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        # x = tf.image.resize(x, size=(shape0[1], shape0[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.image.resize(x, size=(shape1[1], shape1[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.image.resize(x, size=(shape2[1], shape2[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = tf.layers.conv2d(x, filters=3, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        
        return x
    
    def encode_new(self, imgs):
        x = tf.layers.conv2d(imgs, filters=32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        shape2 = x.shape
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
        x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        shape1 = x.shape
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
        shape_orig = x.shape
        x = tf.layers.flatten(x)
        shape_dense = x.shape
        x = tf.layers.dense(x, units=256, activation=tf.nn.relu)
        return x, shape_dense, shape_orig, shape1, shape2
    
    def decode_new(self, encoded, shape_dense, shape_orig, shape1, shape2):
        # Restore from 256 -> 16 * 16 * 64
        x = tf.layers.dense(encoded, units=shape_dense[-1], activation=tf.nn.relu)
        # Restore from 16 * 16 * 64 -> 16x16x64
        x = tf.reshape(x, [-1, shape_orig[1], shape_orig[2], shape_orig[3]])
        # Conv and upsample layers
        x = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.image.resize(x, size=(shape1[1], shape1[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.image.resize(x, size=(shape2[1], shape2[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = tf.layers.conv2d(x, filters=3, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        return x