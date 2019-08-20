# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Wrapper which applies quantization operations over underlying layer.

   `QuantizeWrapper` is responsible for modifying the construction of the
   underlying layer to ensure proper quantization operations are placed in the
   graph.

   These operations ensure proper introduction of inference time losses during
   training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras.layers import deserialize as deserialize_layer
from tensorflow.python.keras.layers.wrappers import Wrapper
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object

from tensorflow_model_optimization.python.core.quantization.keras import quantize_aware_activation


class QuantizeWrapper(Wrapper):
  """Quantizes the weights and activations of the keras layer it wraps."""

  def __init__(self, layer, quantize_provider, **kwargs):
    """Create a quantize emulate wrapper for a keras layer.

    Args:
      layer: The keras layer to be quantized.
      quantize_provider: `QuantizeProvider` to quantize layer.
      **kwargs: Additional keyword arguments to be passed to the keras layer.
    """

    if quantize_provider is None:
      raise ValueError('quantize_provider cannot be None. It is needed to '
                       'quantize a layer.')

    super(QuantizeWrapper, self).__init__(layer, **kwargs)
    self.quantize_provider = quantize_provider

    # Ensures cloning of already built layer works.
    if (not hasattr(self, '_batch_input_shape') and
        hasattr(layer, '_batch_input_shape')):
      self._batch_input_shape = self.layer._batch_input_shape  # pylint: disable=protected-access
    self._track_trackable(layer, name='layer')

  @staticmethod
  def _weight_name(name):
    """Extracts the weight name from the full TensorFlow variable name.

    For example, returns 'kernel' for 'dense_2/kernel:0'.

    Args:
      name: TensorFlow variable name.

    Returns:
      Extracted weight name.
    """
    return name.split(':')[0].split('/')[-1]

  def _add_range_weights(self, name):
    min_weight = self.add_weight(
        name + '_min', initializer=initializers.Constant(-6.0), trainable=False)
    max_weight = self.add_weight(
        name + '_max', initializer=initializers.Constant(6.0), trainable=False)

    return min_weight, max_weight

  def build(self, input_shape):
    super(QuantizeWrapper, self).build(input_shape)

    self.optimizer_step = self.add_weight(
        'optimizer_step',
        initializer=initializers.Constant(-1),
        dtype=dtypes.int32,
        trainable=False)

    self._weight_vars = []
    for weight, quantizer in \
        self.quantize_provider.get_weights_and_quantizers(self.layer):
      min_var, max_var = self._add_range_weights(self._weight_name(weight.name))

      self._weight_vars.append((weight, quantizer, min_var, max_var))
      # Needed to ensure unquantized weights get trained as part of the wrapper.
      self._trainable_weights.append(weight)

    self._quantize_activations = []
    for activation, quantizer in \
        self.quantize_provider.get_activations_and_quantizers(self.layer):
      quantize_activation = quantize_aware_activation.QuantizeAwareActivation(
          activation, quantizer, self.optimizer_step, self)

      self._quantize_activations.append(quantize_activation)

  def compute_output_shape(self, input_shape):
    return self.layer.compute_output_shape(self.layer.input_shape)

  def _dict_vars(self, min_var, max_var):
    return {'min_var': min_var, 'max_var': max_var}

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()

    # Quantize all weights, and replace them in the underlying layer.

    quantized_weights = []
    for unquantized_weight, quantizer, min_var, max_var in self._weight_vars:

      def make_quantizer_fn(training):
        """Use currying to return True/False specialized fns to the cond."""

        def quantizer_fn(unquantized_weight=unquantized_weight,
                         quantizer=quantizer,
                         min_var=min_var,
                         max_var=max_var):
          return quantizer(unquantized_weight, self.optimizer_step, training,
                           **self._dict_vars(min_var, max_var))

        return quantizer_fn

      quantized_weight = tf_utils.smart_cond(
          training, make_quantizer_fn(True), make_quantizer_fn(False))
      quantized_weights.append(quantized_weight)

    self.quantize_provider.set_quantize_weights(self.layer, quantized_weights)

    # Replace all activations with `QuantizeAwareActivation`s which can
    # quantize activation tensors during graph construction.

    for quantize_activation in self._quantize_activations:
      quantize_activation.training = training

    self.quantize_provider.set_quantize_activations(
        self.layer, self._quantize_activations)

    return self.layer.call(inputs)

  def get_config(self):
    base_config = super(QuantizeWrapper, self).get_config()
    config = {
        'quantize_provider': serialize_keras_object(self.quantize_provider)
    }
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    config = config.copy()

    # QuantizeWrapper may be constructed with any QuantizeProvider and the
    # wrapper itself cannot know all the possible provider classes.
    # The deserialization code should ensure the QuantizeProvider is in keras
    # serialization scope.
    quantize_provider = deserialize_keras_object(
        config.pop('quantize_provider'),
        module_objects=globals(),
        custom_objects=None)

    layer = deserialize_layer(config.pop('layer'))

    return cls(layer=layer, quantize_provider=quantize_provider, **config)

  @property
  def trainable(self):
    return self.layer.trainable

  @trainable.setter
  def trainable(self, value):
    self.layer.trainable = value

  @property
  def trainable_weights(self):
    return self.layer.trainable_weights + self._trainable_weights

  @property
  def non_trainable_weights(self):
    return self.layer.non_trainable_weights + self._non_trainable_weights

  @property
  def updates(self):
    return self.layer.updates + self._updates

  @property
  def losses(self):
    return self.layer.losses + self._losses
