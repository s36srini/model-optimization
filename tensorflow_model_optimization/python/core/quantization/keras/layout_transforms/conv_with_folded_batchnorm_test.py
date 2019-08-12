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
"""ConvWithFoldedBatchNorm layout transformation tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf

from tensorflow.python import keras
from tensorflow.python.platform import test
from tensorflow_model_optimization.python.core.quantization.keras.layout_transforms.conv_with_folded_batchnorm import ConvWithFoldedBatchNorm
from tensorflow_model_optimization.python.core.quantization.keras.utils import convert_mnist_to_tflite


class ConvWithFoldedBatchNormTest(test.TestCase):

  def setUp(self):
    super(ConvWithFoldedBatchNormTest, self).setUp()
    self.model_params = {
        'filters': 2,
        'kernel_size': (3, 3),
        'input_shape': (10, 10, 3),
    }

  def _get_folded_batchnorm_model(self):
    return tf.keras.Sequential([
        ConvWithFoldedBatchNorm(
            self.model_params['filters'],
            self.model_params['kernel_size'],
            input_shape=self.model_params['input_shape'],
            kernel_initializer=keras.initializers.glorot_uniform(seed=0))
    ])

  def testEquivalentToNonFoldedBatchNorm(self):
    model_1 = self._get_folded_batchnorm_model()

    model_2 = tf.keras.Sequential([
        keras.layers.Conv2D(
            self.model_params['filters'],
            self.model_params['kernel_size'],
            input_shape=self.model_params['input_shape'],
            kernel_initializer=keras.initializers.glorot_uniform(seed=0),
            use_bias=False),
        keras.layers.BatchNormalization(axis=-1),
    ])

    for _ in range(10):
      inp = np.random.randint(10, size=[1, 10, 10, 3])
      out = model_1.predict(inp)
      out2 = model_2.predict(inp)

    # Taken from testFoldFusedBatchNorms from
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference_test.py#L230
    np.testing.assert_allclose(out, out2, rtol=1e-04, atol=1e-06)

  def testEquivalentToTFLite(self):
    model = self._get_folded_batchnorm_model()

    _, keras_file = tempfile.mkstemp('.h5')
    _, tflite_file = tempfile.mkstemp('.h5')

    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    model.fit(
        # normalize to 0 to 1.
        np.random.randint(255, size=[1, 10, 10, 3]) / 255,
        np.random.randint(10, size=[1, 8, 8, 2]),
        epochs=1,
        callbacks=[])

    # Prepare for inference.
    inp = np.random.randint(255, size=[1, 10, 10, 3]) / 255.0
    inp = inp.astype(np.float32)

    # TensorFlow inference.
    out = model.predict(inp)

    # TensorFlow Lite inference.
    tf.keras.models.save_model(model, keras_file)
    convert_mnist_to_tflite(
        keras_file,
        tflite_file,
        custom_objects={'ConvWithFoldedBatchNorm': ConvWithFoldedBatchNorm},
        is_quantized=False)

    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    interpreter.set_tensor(input_index, inp)
    interpreter.invoke()
    out2 = interpreter.get_tensor(output_index)

    # Equality check.
    np.testing.assert_allclose(out, out2, rtol=1e-04, atol=1e-06)


if __name__ == '__main__':
  test.main()
