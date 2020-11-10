# Lint as: python3
# Copyright Hassan Akbari 2020. All Rights Reserved.
# ha2436@columbia.edu
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
# ==============================================================================
"""Model defination for the I3D Video Model."""

import tensorflow as tf


class Unit3D(tf.keras.layers.Layer):
  """The main 3D unit that contains Conv3D + BN."""

  def __init__(self, output_channels,
               kernel_size=(1, 1, 1),
               strides=(1, 1, 1),
               activation_fn=tf.nn.relu,
               use_batch_norm=True,
               use_bias=False,
               use_xreplica_bn=True,
               name='unit_3d',
               **kwargs):
    super(Unit3D, self).__init__(name=name, **kwargs)
    self._use_batch_norm = use_batch_norm
    self._activation_fn = activation_fn
    self.conv3d = tf.keras.layers.Conv3D(
        filters=output_channels,
        kernel_size=kernel_size,
        strides=strides,
        use_bias=use_bias,
        padding='same',
        name='conv_3d'
        )

    if use_xreplica_bn:
      bn_fn = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      bn_fn = tf.keras.layers.experimental.BatchNormalization
    self.bn = bn_fn(scale=False, name='batch_norm')

  def call(self, inputs, training=None):
    out = self.conv3d(inputs)
    if self._use_batch_norm:
      out = self.bn(out, training=training)
    if self._activation_fn is not None:
      out = self._activation_fn(out)
    return out


class Mixed(tf.keras.layers.Layer):
  """The 3D Inception block."""

  def __init__(self,
               filter_map,
               use_xreplica_bn,
               name='mixed',
               **kwargs):
    super(Mixed, self).__init__(name=name, **kwargs)
    self.unit3d_b0_0 = Unit3D(output_channels=filter_map['branch_0'],
                              kernel_size=[1, 1, 1],
                              use_xreplica_bn=use_xreplica_bn,
                              name='conv3d_0a_1x1')
    self.unit3d_b1_0 = Unit3D(output_channels=filter_map['branch_1'][0],
                              kernel_size=[1, 1, 1],
                              use_xreplica_bn=use_xreplica_bn,
                              name='conv3d_0a_1x1')
    self.unit3d_b1_1 = Unit3D(output_channels=filter_map['branch_1'][1],
                              kernel_size=[3, 3, 3],
                              use_xreplica_bn=use_xreplica_bn,
                              name='conv3d_0b_3x3')
    self.unit3d_b2_0 = Unit3D(output_channels=filter_map['branch_2'][0],
                              kernel_size=[1, 1, 1],
                              use_xreplica_bn=use_xreplica_bn,
                              name='conv3d_0a_1x1')
    self.unit3d_b2_1 = Unit3D(output_channels=filter_map['branch_2'][1],
                              kernel_size=[3, 3, 3],
                              use_xreplica_bn=use_xreplica_bn,
                              name='conv3d_0b_3x3')
    self.max3d_b3_0 = tf.keras.layers.MaxPool3D(pool_size=[3, 3, 3],
                                                strides=[1, 1, 1],
                                                padding='same',
                                                name='maxpool3d_0a_3x3')
    self.unit3d_b3_1 = Unit3D(output_channels=filter_map['branch_3'],
                              kernel_size=[1, 1, 1],
                              use_xreplica_bn=use_xreplica_bn,
                              name='conv3d_0b_1x1')

  def call(self, inputs, training=None):
    with tf.name_scope('branch_0'):
      branch_0 = self.unit3d_b0_0(inputs, training=training)

    with tf.name_scope('branch_1'):
      branch_1 = self.unit3d_b1_0(inputs, training=training)
      branch_1 = self.unit3d_b1_1(branch_1, training=training)

    with tf.name_scope('branch_2'):
      branch_2 = self.unit3d_b2_0(inputs, training=training)
      branch_2 = self.unit3d_b2_1(branch_2, training=training)

    with tf.name_scope('branch_3'):
      branch_3 = self.max3d_b3_0(inputs)
      branch_3 = self.unit3d_b3_1(branch_3, training=training)

    output = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    return output


class InceptionI3D(tf.keras.layers.Layer):
  """Inception I3D Model."""
  _FEATURE_LAYERS = [
      'conv3d_1a_7x7',
      'maxpool3d_2a_3x3',
      'conv3d_2b_1x1',
      'conv3d_2c_3x3',
      'maxpool3d_3a_3x3',
      'mixed_3b',
      'mixed_3c',
      'maxpool3d_4a_3x3',
      'mixed_4b',
      'mixed_4c',
      'mixed_4d',
      'mixed_4e',
      'mixed_4f',
      'maxpool3d_5a_2x2',
      'mixed_5b',
      'mixed_5c'
  ]

  def __init__(self,
               num_classes=None,
               use_xreplica_bn=True,
               dropout_rate=0.2,
               name='i3d_backbone',
               **kwargs):

    super(InceptionI3D, self).__init__(name=name)
    self.num_classes = num_classes
    self.conv3d_1a_7x7 = Unit3D(output_channels=64,
                                kernel_size=[7, 7, 7],
                                strides=[2, 2, 2],
                                use_xreplica_bn=use_xreplica_bn,
                                name='conv3d_1a_7x7')
    self.maxpool3d_2a_3x3 = tf.keras.layers.MaxPool3D(pool_size=[1, 3, 3],
                                                      strides=[1, 2, 2],
                                                      padding='same',
                                                      name='maxpool3d_2a_3x3')
    self.conv3d_2b_1x1 = Unit3D(output_channels=64,
                                kernel_size=[1, 1, 1],
                                use_xreplica_bn=use_xreplica_bn,
                                name='conv3d_2b_1x1')
    self.conv3d_2c_3x3 = Unit3D(output_channels=192,
                                kernel_size=[3, 3, 3],
                                use_xreplica_bn=use_xreplica_bn,
                                name='conv3d_2c_3x3')
    self.maxpool3d_3a_3x3 = tf.keras.layers.MaxPool3D(pool_size=[1, 3, 3],
                                                      strides=[1, 2, 2],
                                                      padding='same',
                                                      name='maxpool3d_3a_3x3')
    self.mixed_3b = Mixed(filter_map={'branch_0': 64,
                                      'branch_1': [96, 128],
                                      'branch_2': [16, 32],
                                      'branch_3': 32},
                          use_xreplica_bn=use_xreplica_bn,
                          name='mixed_3b')
    self.mixed_3c = Mixed(filter_map={'branch_0': 128,
                                      'branch_1': [128, 192],
                                      'branch_2': [32, 96],
                                      'branch_3': 64},
                          use_xreplica_bn=use_xreplica_bn,
                          name='mixed_3c')
    self.maxpool3d_4a_3x3 = tf.keras.layers.MaxPool3D(pool_size=[3, 3, 3],
                                                      strides=[2, 2, 2],
                                                      padding='same',
                                                      name='maxpool3d_4a_3x3')
    self.mixed_4b = Mixed(filter_map={'branch_0': 192,
                                      'branch_1': [96, 208],
                                      'branch_2': [16, 48],
                                      'branch_3': 64},
                          use_xreplica_bn=use_xreplica_bn,
                          name='mixed_4b')
    self.mixed_4c = Mixed(filter_map={'branch_0': 160,
                                      'branch_1': [112, 224],
                                      'branch_2': [24, 64],
                                      'branch_3': 64},
                          use_xreplica_bn=use_xreplica_bn,
                          name='mixed_4c')
    self.mixed_4d = Mixed(filter_map={'branch_0': 128,
                                      'branch_1': [128, 256],
                                      'branch_2': [24, 64],
                                      'branch_3': 64},
                          use_xreplica_bn=use_xreplica_bn,
                          name='mixed_4d')
    self.mixed_4e = Mixed(filter_map={'branch_0': 112,
                                      'branch_1': [144, 288],
                                      'branch_2': [32, 64],
                                      'branch_3': 64},
                          use_xreplica_bn=use_xreplica_bn,
                          name='mixed_4e')
    self.mixed_4f = Mixed(filter_map={'branch_0': 256,
                                      'branch_1': [160, 320],
                                      'branch_2': [32, 128],
                                      'branch_3': 128},
                          use_xreplica_bn=use_xreplica_bn,
                          name='mixed_4f')
    self.maxpool3d_5a_2x2 = tf.keras.layers.MaxPool3D(pool_size=[2, 2, 2],
                                                      strides=[2, 2, 2],
                                                      padding='same',
                                                      name='maxpool3d_5a_2x2')
    self.mixed_5b = Mixed(filter_map={'branch_0': 256,
                                      'branch_1': [160, 320],
                                      'branch_2': [32, 128],
                                      'branch_3': 128},
                          use_xreplica_bn=use_xreplica_bn,
                          name='mixed_5b')
    self.mixed_5c = Mixed(filter_map={'branch_0': 384,
                                      'branch_1': [192, 384],
                                      'branch_2': [48, 128],
                                      'branch_3': 128},
                          use_xreplica_bn=use_xreplica_bn,
                          name='mixed_5c')
    if self.num_classes is not None:
      self.dropout_la = tf.keras.layers.Dropout(dropout_rate, name='dropout')
      self.conv3d_0c_1x1 = Unit3D(output_channels=num_classes,
                                  kernel_size=[1, 1, 1],
                                  activation_fn=None,
                                  use_batch_norm=False,
                                  use_bias=True,
                                  use_xreplica_bn=use_xreplica_bn,
                                  name='conv3d_0c_1x1')

  def _add_avg_pool3d(self, t, h, w):
    self.avgpool3d_la_txhxw = tf.keras.layers.AveragePooling3D(
        pool_size=[t, h, w],
        strides=[1, 1, 1],
        padding='valid',
        name='avgpool3d_la_txhxw'
        )

  def freeze_backbone(self):
    for layer_name in self._FEATURE_LAYERS:
      layer = getattr(self, layer_name)
      layer.trainable = False

  def unfreeze_backbone(self):
    for layer_name in self._FEATURE_LAYERS:
      layer = getattr(self, layer_name)
      layer.trainable = True

  def freeze_classification_layer(self):
    self.conv3d_0c_1x1.trainable = False

  def unfreeze_classification_layer(self):
    self.conv3d_0c_1x1.trainable = True

  def call(self,
           inputs,
           training):

    endpoints = {}
    for layer_name in self._FEATURE_LAYERS:
      layer = getattr(self, layer_name)
      if 'maxpool3d' in layer_name:
        inputs = layer(inputs)
      else:
        inputs = layer(inputs, training)
      endpoints[layer_name] = inputs

    if self.num_classes is None:
      return inputs, endpoints
    else:
      if not hasattr(self, 'avgpool3d_la_txhxw'):
        t, h, w = inputs.shape[1:4]
        self._add_avg_pool3d(t, h, w)

      with tf.name_scope('logits'):
        inputs = self.avgpool3d_la_txhxw(inputs)
        inputs = self.dropout_la(inputs, training)

        logits = self.conv3d_0c_1x1(inputs, training)
        logits = tf.squeeze(logits, [2, 3], name='spatial_squeeze')

        averaged_logits = tf.reduce_mean(logits, axis=1)
        predictions = tf.nn.softmax(averaged_logits)

      return predictions
