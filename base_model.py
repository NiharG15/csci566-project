# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base Music Variational Autoencoder (MusicVAE) model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf
import tensorflow_probability as tfp


ds = tfp.distributions

import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
from tensorflow.contrib.slim import conv2d
import matplotlib.pyplot as plt 

from image_vae import Autoencoder



class BaseEncoder(object):
  """Abstract encoder class.

    Implementations must define the following abstract methods:
     -`build`
     -`encode`
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def output_depth(self):
    """Returns the size of the output final dimension."""
    pass

  @abc.abstractmethod
  def build(self, hparams, is_training=True):
    """Builder method for BaseEncoder.

    Args:
      hparams: An HParams object containing model hyperparameters.
      is_training: Whether or not the model is being used for training.
    """
    pass

  @abc.abstractmethod
  def encode(self, sequence, sequence_length):
    """Encodes input sequences into a precursors for latent code `z`.

    Args:
       sequence: Batch of sequences to encode.
       sequence_length: Length of sequences in input batch.

    Returns:
       outputs: Raw outputs to parameterize the prior distribution in
          MusicVae.encode, sized `[batch_size, N]`.
    """
    pass


class BaseDecoder(object):
  """Abstract decoder class.

  Implementations must define the following abstract methods:
     -`build`
     -`reconstruction_loss`
     -`sample`
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def build(self, hparams, output_depth, is_training=True):
    """Builder method for BaseDecoder.

    Args:
      hparams: An HParams object containing model hyperparameters.
      output_depth: Size of final output dimension.
      is_training: Whether or not the model is being used for training.
    """
    pass

  @abc.abstractmethod
  def reconstruction_loss(self, x_input, x_target, x_length, z=None,
                          c_input=None):
    """Reconstruction loss calculation.

    Args:
      x_input: Batch of decoder input sequences for teacher forcing, sized
          `[batch_size, max(x_length), output_depth]`.
      x_target: Batch of expected output sequences to compute loss against,
          sized `[batch_size, max(x_length), output_depth]`.
      x_length: Length of input/output sequences, sized `[batch_size]`.
      z: (Optional) Latent vectors. Required if model is conditional. Sized
          `[n, z_size]`.
      c_input: (Optional) Batch of control sequences, sized
          `[batch_size, max(x_length), control_depth]`. Required if conditioning
          on control sequences.

    Returns:
      r_loss: The reconstruction loss for each sequence in the batch.
      metric_map: Map from metric name to tf.metrics return values for logging.
    """
    pass

  @abc.abstractmethod
  def sample(self, n, max_length=None, z=None, c_input=None):
    """Sample from decoder with an optional conditional latent vector `z`.

    Args:
      n: Scalar number of samples to return.
      max_length: (Optional) Scalar maximum sample length to return. Required if
        data representation does not include end tokens.
      z: (Optional) Latent vectors to sample from. Required if model is
        conditional. Sized `[n, z_size]`.
      c_input: (Optional) Control sequence, sized `[max_length, control_depth]`.

    Returns:
      samples: Sampled sequences. Sized `[n, max_length, output_depth]`.
    """
    pass


class MusicVAE(object):
  """Music Variational Autoencoder."""

  def __init__(self, encoder, decoder):
    """Initializer for a MusicVAE model.

    Args:
      encoder: A BaseEncoder implementation class to use.
      decoder: A BaseDecoder implementation class to use.
    """
    self._encoder = encoder
    self._decoder = decoder
    

    

  def build(self, hparams, output_depth, is_training):
    """Builds encoder and decoder.

    Must be called within a graph.

    Args:
      hparams: An HParams object containing model hyperparameters. See
          `get_default_hparams` below for required values.
      output_depth: Size of final output dimension.
      is_training: Whether or not the model will be used for training.
    """
    
    tf.logging.info('Building MusicVAE model with %s, %s, and hparams:\n%s',
                    self.encoder.__class__.__name__,
                    self.decoder.__class__.__name__,
                    hparams.values())
    self.global_step = tf.train.get_or_create_global_step()
    self._hparams = hparams
    self._encoder.build(hparams, is_training)
    self._decoder.build(hparams, output_depth, is_training)
    self.ae = Autoencoder(n_z=hparams.z_size)
    self.ae.build_var()
    self.shared_z = tf.layers.Dense(units=hparams.z_size, activation=tf.nn.leaky_relu, name='shared_z')
    self.shared_z2 = tf.layers.Dense(units=hparams.z_size, activation=tf.nn.leaky_relu, name='shared_z2')

  @property
  def encoder(self):
    return self._encoder

  @property
  def decoder(self):
    return self._decoder

  @property
  def hparams(self):
    return self._hparams

  def encode(self, sequence, sequence_length, control_sequence=None):
    """Encodes input sequences into a MultivariateNormalDiag distribution.

    Args:
      sequence: A Tensor with shape `[num_sequences, max_length, input_depth]`
          containing the sequences to encode.
      sequence_length: The length of each sequence in the `sequence` Tensor.
      control_sequence: (Optional) A Tensor with shape
          `[num_sequences, max_length, control_depth]` containing control
          sequences on which to condition. These will be concatenated depthwise
          to the input sequences.

    Returns:
      A tfp.distributions.MultivariateNormalDiag representing the posterior
      distribution for each sequence.
    """
    

    hparams = self.hparams
    z_size = hparams.z_size

    sequence = tf.to_float(sequence)
    # with tf.Session() as sess:  print('dadadadada',sequence.eval(),sequence,sequence.shape) 
    if control_sequence is not None:
      control_sequence = tf.to_float(control_sequence)
      sequence = tf.concat([sequence, control_sequence], axis=-1)
    print('rararara',self.encoder)
    encoder_output = self.encoder.encode(sequence, sequence_length)

    with tf.variable_scope('encoder/qz', reuse=tf.AUTO_REUSE):
        mu = tf.layers.dense(
            encoder_output,
            z_size,
            name='encoder/mu',
            kernel_initializer=tf.random_normal_initializer(stddev=0.001))
        sigma = tf.layers.dense(
            encoder_output,
            z_size,
            activation=tf.nn.softplus,
            name='encoder/sigma',
            kernel_initializer=tf.random_normal_initializer(stddev=0.001))

    # eps = tf.random_normal(
    #         shape=tf.shape(sigma),
    #         mean=0, stddev=1, dtype=tf.float32)

    # return mu,sigma# + tf.sqrt(tf.exp(sigma)) * eps
    return ds.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

  def _compute_model_loss(
      self, input_sequence, output_sequence, sequence_length, control_sequence, image_input, summary=True):
    """Builds a model with loss for train/eval."""
    hparams = self.hparams
    batch_size = hparams.batch_size

    input_sequence = tf.to_float(input_sequence)
    output_sequence = tf.to_float(output_sequence)

    max_seq_len = tf.minimum(tf.shape(output_sequence)[1], hparams.max_seq_len)

    input_sequence = input_sequence[:, :max_seq_len]

    if control_sequence is not None:
      control_depth = control_sequence.shape[-1]
      control_sequence = tf.to_float(control_sequence)
      control_sequence = control_sequence[:, :max_seq_len]
      # Shouldn't be necessary, but the slice loses shape information when
      # control depth is zero.
      control_sequence.set_shape([batch_size, None, control_depth])

    # The target/expected outputs.
    x_target = output_sequence[:, :max_seq_len]
    # Inputs to be fed to decoder, including zero padding for the initial input.
    x_input = tf.pad(output_sequence[:, :max_seq_len - 1],
                     [(0, 0), (1, 0), (0, 0)])
    x_length = tf.minimum(sequence_length, max_seq_len)

    # Either encode to get `z`, or do unconditional, decoder-only.
    if hparams.z_size:  # vae mode:
      # print('lalalala', input_sequence)
      music_qz = self.encode(input_sequence, x_length, control_sequence)
      # print(mu_music,sigma_music)
      image_dist, *shapes = self.ae.encode_var_new(image_input)
      mu_image = image_dist.loc
      sigma_image = image_dist.scale.diag


      # 1. Pass Music through encoder
      # 2. Pass image through encoder
      # 3. Pass result of image encoder through music decoder
      # 4. Pass result of music encoder through image decoder
      # TODO: Try adding image to image and music to music loss.


      # factor = hparams.batch_size // hparams.image_batch_size
      # mu_image_final = tf.tile(mu_image, [factor, 1])
      # sigma_image_final = tf.tile(sigma_image, [factor, 1])

      if summary:
        tf.summary.image('input_midi', tf.expand_dims(input_sequence, axis=-1), max_outputs=5)
        tf.summary.image('input_images', image_input, max_outputs=5)
      mu_music, sigma_music = music_qz.loc, music_qz.scale.diag
      
      
      music_z = music_qz.sample()
      image_z = image_dist.sample()

      # print(music_z.shape)
      # print(image_z.shape)
      # print(x_input.shape, x_target.shape, sequence_length.shape)

      # Prior distribution.
      p_z = ds.MultivariateNormalDiag(
          loc=[0.] * hparams.z_size, scale_diag=[1.] * hparams.z_size)

      # KL Divergence (nats)
      kl_div = ds.kl_divergence(music_qz, p_z)
      kl_div_img = ds.kl_divergence(image_dist, p_z)
      # Concatenate the Z vectors to the inputs at each time step.
    else:  # unconditional, decoder-only generation
      kl_div = tf.zeros([batch_size, 1], dtype=tf.float32)
      z = None

    stacked_z = tf.concat((image_z, music_z), axis=0)
    # print(stacked_z.shape)
    stacked_z_out = self.shared_z2(self.shared_z(stacked_z))
    # print(stacked_z_out.shape)
    image_z, music_z = tf.split(stacked_z_out, num_or_size_splits=2, axis=0)
    # print(music_z.shape)
    # print(image_z.shape)
    r_loss, metric_map = self.decoder.reconstruction_loss(
        x_input, x_target, x_length, image_z, control_sequence)[0:2]

    # r_loss_music, metric_map = self.decoder.reconstruction_loss(
    #     x_input, x_target, x_length, music_z, control_sequence)[0:2]

    ## Image Reconstruction Loss through music
    recon_imgs = self.ae.decode_var_new(music_z, *shapes)
    if summary:
      tf.summary.image('recon_images', recon_imgs, max_outputs=5)
    flat_imgs = tf.reshape(image_input, (tf.shape(image_input)[0], -1))
    flat_recon_imgs = tf.reshape(recon_imgs, (tf.shape(recon_imgs)[0], -1))

    recon_loss = tf.reduce_mean(tf.pow(flat_imgs - flat_recon_imgs, 2))
    recon_loss *= 1600

    # recon_imgs_imgs = self.ae.decode_var_new(image_z, *shapes)
    # if summary:
    #   tf.summary.image('recon_imgs_imgs', recon_imgs_imgs, max_outputs=5)
    
    # flat_recon_imgs_imgs = tf.reshape(recon_imgs_imgs, (tf.shape(recon_imgs_imgs)[0], -1))

    # recon_loss_imgs = tf.reduce_sum(tf.pow(flat_imgs - flat_recon_imgs_imgs, 2), axis=1)

    
    gamma = hparams.gamma

    random_z = tf.random.normal(music_z.shape)
    random_music = self.decoder.reconstruction_loss(x_input, x_target, x_length, random_z, control_sequence)[2].rnn_output
    random_image = self.ae.decode_var_new(random_z, *shapes)

    random_music = tf.stop_gradient(random_music)
    random_image = tf.stop_gradient(random_image)

    reverse_music_dist = self.encode(random_music, x_length, control_sequence)
    reverse_image_dist, *reverse_shapes = self.ae.encode_var_new(random_image)
    reverse_music_mu = reverse_music_dist.loc
    reverse_image_mu = reverse_image_dist.loc
    reverse_stacked_mu = tf.concat((reverse_image_mu, reverse_music_mu), axis=0)
    reverse_image_z, reverse_music_z = tf.split(self.shared_z(reverse_stacked_mu), num_or_size_splits=2, axis=0)

    reverse_cycle_loss = tf.reduce_mean(tf.abs(reverse_image_z - reverse_music_z))
    reverse_cycle_loss *= 1600
    

    free_nats = hparams.free_bits * tf.math.log(2.0)
    kl_cost = tf.maximum(kl_div - free_nats, 0)
    kl_cost_img = tf.maximum(kl_div_img - free_nats, 0)

    beta = ((1.0 - tf.pow(hparams.beta_rate, tf.to_float(self.global_step)))
            * hparams.max_beta)

    self.loss = tf.reduce_mean(r_loss) + recon_loss + beta * (tf.reduce_mean(kl_cost) + tf.reduce_mean(kl_div_img))
    
    if gamma > 0:
        self.loss += gamma * reverse_cycle_loss

    scalars_to_summarize = {
        'loss': self.loss,
        'losses/r_loss': r_loss,
        'losses/kl_loss': kl_cost,
        'losses/kl_bits': kl_div / tf.math.log(2.0),
        'losses/kl_beta': beta,
        'losses/image_recon_loss': recon_loss,
        'losses/image_kl_div': kl_div_img,
        'losses/reverse_cycle_loss': reverse_cycle_loss,
        # 'loss/recon_loss_img_img': recon_loss_imgs,
        # 'losses/r_loss_music': r_loss_music
    }
    return metric_map, scalars_to_summarize

  def train(self, input_sequence, output_sequence, sequence_length,image_input,
            control_sequence=None):
    """Train on the given sequences, returning an optimizer.

    Args:
      input_sequence: The sequence to be fed to the encoder.
      output_sequence: The sequence expected from the decoder.
      sequence_length: The length of the given sequences (which must be
          identical).
      control_sequence: (Optional) sequence on which to condition. This will be
          concatenated depthwise to the model inputs for both encoding and
          decoding.

    Returns:
      optimizer: A tf.train.Optimizer.
    """
    # with tf.Session() as sess:  print('dadadadada',input_sequence.eval(),input_sequence,input_sequence.shape) 
    

    _, scalars_to_summarize = self._compute_model_loss(
        input_sequence, output_sequence, sequence_length, control_sequence,image_input)

    hparams = self.hparams
    lr = ((hparams.learning_rate - hparams.min_learning_rate) *
          tf.pow(hparams.decay_rate, tf.to_float(self.global_step)) +
          hparams.min_learning_rate)

    optimizer = tf.train.AdamOptimizer(lr)

    tf.summary.scalar('learning_rate', lr)
    for n, t in scalars_to_summarize.items():
      tf.summary.scalar(n, tf.reduce_mean(t))

    return optimizer

  def eval(self, input_sequence, output_sequence, sequence_length,
           control_sequence=None):
    """Evaluate on the given sequences, returning metric update ops.

    Args:
      input_sequence: The sequence to be fed to the encoder.
      output_sequence: The sequence expected from the decoder.
      sequence_length: The length of the given sequences (which must be
        identical).
      control_sequence: (Optional) sequence on which to condition the decoder.

    Returns:
      metric_update_ops: tf.metrics update ops.
    """
    metric_map, scalars_to_summarize = self._compute_model_loss(
        input_sequence, output_sequence, sequence_length, control_sequence)

    for n, t in scalars_to_summarize.iteritems():
      metric_map[n] = tf.metrics.mean(t)

    metrics_to_values, metrics_to_updates = (
        tf.contrib.metrics.aggregate_metric_map(metric_map))

    for metric_name, metric_value in metrics_to_values.iteritems():
      tf.summary.scalar(metric_name, metric_value)

    return metrics_to_updates.values()

  def sample(self, n, max_length=None, z=None, c_input=None, **kwargs):
    """Sample with an optional conditional embedding `z`."""
    if z is not None and z.shape[0].value != n:
      raise ValueError(
          '`z` must have a first dimension that equals `n` when given. '
          'Got: %d vs %d' % (z.shape[0].value, n))

    if self.hparams.z_size and z is None:
      tf.logging.warning(
          'Sampling from conditional model without `z`. Using random `z`.')
      normal_shape = [n, self.hparams.z_size]
      normal_dist = tfp.distributions.Normal(
          loc=tf.zeros(normal_shape), scale=tf.ones(normal_shape))
      z = normal_dist.sample()

    return self.decoder.sample(n, max_length, z, c_input, **kwargs)


def get_default_hparams():
  return tf.contrib.training.HParams(
      max_seq_len=32,  # Maximum sequence length. Others will be truncated.
      z_size=32,  # Size of latent vector z.
      free_bits=0.0,  # Bits to exclude from KL loss per dimension.
      max_beta=1.0,  # Maximum KL cost weight, or cost if not annealing.
      beta_rate=0.0,  # Exponential rate at which to anneal KL cost.
      gamma=5, # Coefficient for reverse cycle loss.
      batch_size=512,  # Minibatch size.
      grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.
      clip_mode='global_norm',  # value or global_norm.
      # If clip_mode=global_norm and global_norm is greater than this value,
      # the gradient will be clipped to 0, effectively ignoring the step.
      grad_norm_clip_to_zero=10000,
      learning_rate=0.0001,  # Learning rate.
      decay_rate=0.9999,  # Learning rate decay per minibatch.
      min_learning_rate=0.000001,  # Minimum learning rate.
  )
