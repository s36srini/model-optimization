# Copyright 2019, The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import mock
import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow as tf

from tensorflow_model_optimization.python.core.internal.tensor_encoding.core import encoding_stage
from tensorflow_model_optimization.python.core.internal.tensor_encoding.testing import test_utils


class PlusOneEncodingStageTest(test_utils.BaseEncodingStageTest):

  def default_encoding_stage(self):
    """See base class."""
    return test_utils.PlusOneEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.random.uniform([5])

  @property
  def is_lossless(self):
    """See base class."""
    return True

  def common_asserts_for_test_data(self, data):
    """See base class."""
    self.assertAllClose(data.x, data.decoded_x)
    self.assertAllClose(
        data.x + 1.0,
        data.encoded_x[test_utils.PlusOneEncodingStage.ENCODED_VALUES_KEY])


class TimesTwoEncodingStageTest(test_utils.BaseEncodingStageTest):

  def default_encoding_stage(self):
    """See base class."""
    return test_utils.TimesTwoEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.random.uniform([5])

  @property
  def is_lossless(self):
    """See base class."""
    return True

  def common_asserts_for_test_data(self, data):
    """See base class."""
    self.assertAllClose(data.x, data.decoded_x)
    self.assertAllClose(
        data.x * 2.0,
        data.encoded_x[test_utils.TimesTwoEncodingStage.ENCODED_VALUES_KEY])


class SimpleLinearEncodingStageTest(test_utils.BaseEncodingStageTest):

  _DEFAULT_A = 2.0
  _DEFAULT_B = 3.0
  _ENCODED_VALUES_KEY = test_utils.SimpleLinearEncodingStage.ENCODED_VALUES_KEY

  def default_encoding_stage(self):
    """See base class."""
    return test_utils.SimpleLinearEncodingStage(self._DEFAULT_A,
                                                self._DEFAULT_B)

  def default_input(self):
    """See base class."""
    return tf.random.uniform([5])

  @property
  def is_lossless(self):
    """See base class."""
    return True

  def common_asserts_for_test_data(self, data):
    """See base class."""
    self.assertAllClose(data.x, data.decoded_x)
    self.assertAllClose(data.x * self._DEFAULT_A + self._DEFAULT_B,
                        data.encoded_x[self._ENCODED_VALUES_KEY])

  def test_basic_encode_decode_tf_constructor_parameters(self):
    """Tests the core funcionality with `tf.Variable` constructor parameters."""
    a_var = tf.get_variable('a_var', initializer=self._DEFAULT_A)
    b_var = tf.get_variable('b_var', initializer=self._DEFAULT_B)
    stage = test_utils.SimpleLinearEncodingStage(a_var, b_var)

    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
    x = self.default_input()
    encode_params, decode_params = stage.get_params()
    encoded_x, decoded_x = self.encode_decode_x(stage, x, encode_params,
                                                decode_params)
    test_data = self.evaluate_test_data(
        test_utils.TestData(x, encoded_x, decoded_x))
    self.common_asserts_for_test_data(test_data)

    # Change the variables and verify the behavior of stage changes.
    self.evaluate([tf.assign(a_var, 5.0), tf.assign(b_var, 6.0)])
    test_data = self.evaluate_test_data(
        test_utils.TestData(x, encoded_x, decoded_x))
    self.assertAllClose(test_data.x * 5.0 + 6.0,
                        test_data.encoded_x[self._ENCODED_VALUES_KEY])


class ReduceMeanEncodingStageTest(test_utils.BaseEncodingStageTest):

  def default_encoding_stage(self):
    """See base class."""
    return test_utils.ReduceMeanEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.random.uniform([5])

  @property
  def is_lossless(self):
    """See base class."""
    return False

  def common_asserts_for_test_data(self, data):
    """See base class."""
    self.assertAllClose(np.tile(np.mean(data.x), data.x.shape), data.decoded_x)
    self.assertAllClose(
        np.mean(data.x, keepdims=True),
        data.encoded_x[test_utils.ReduceMeanEncodingStage.ENCODED_VALUES_KEY])

  def test_one_to_many_with_unknown_shape(self):
    """Tests that encoding works with statically not known input shape."""
    test_data = self.run_one_to_many_encode_decode(
        self.default_encoding_stage(), test_utils.get_tensor_with_random_shape)
    self.common_asserts_for_test_data(test_data)

  @parameterized.parameters([2], [2, 3], [2, 3, 4], [2, 3, 4, 5])
  def test_one_to_many_with_multiple_input_shapes(self, *shape):
    test_data = self.run_one_to_many_encode_decode(
        self.default_encoding_stage(), lambda: tf.random_uniform(shape))
    self.common_asserts_for_test_data(test_data)


class RandomAddSubtractOneEncodingStageTest(test_utils.BaseEncodingStageTest):

  _VALUES_KEY = test_utils.RandomAddSubtractOneEncodingStage.ENCODED_VALUES_KEY

  def default_encoding_stage(self):
    """See base class."""
    return test_utils.RandomAddSubtractOneEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.random.normal([5])

  @property
  def is_lossless(self):
    """See base class."""
    return False

  def common_asserts_for_test_data(self, data):
    """See base class."""
    for x, decoded_x in zip(data.x, data.decoded_x):
      self.assertTrue(
          np.isclose(decoded_x, x - 1) or np.isclose(decoded_x, x) or
          np.isclose(decoded_x, x + 1))
    self.assertAllEqual(data.encoded_x[self._VALUES_KEY], data.decoded_x)

  def test_approximately_unbiased_in_expectation(self):
    """Tests that average of encodings is more accurate than a single one."""
    # Use a constant input value.
    x = self.evaluate(self.default_input())
    stage = self.default_encoding_stage()
    encode_params, decode_params = stage.get_params()
    encoded_x, decoded_x = self.encode_decode_x(stage, x, encode_params,
                                                decode_params)
    test_data = []
    for _ in range(100):
      test_data.append(
          test_utils.TestData(
              *self.evaluate_tf_py_list([x, encoded_x, decoded_x])))

    # Check that the average error created by encoding is significantly larger
    # than error of average of encodings. This is an simple (imperfect)
    # empirical check that the encoding is unbiased.
    mean_error = np.mean([np.linalg.norm(x - d.decoded_x) for d in test_data])
    error_of_mean = np.linalg.norm(
        x - np.mean([d.decoded_x for d in test_data], axis=0))
    self.assertGreater(mean_error, error_of_mean * 5)


class SignIntFloatEncodingStageTest(test_utils.BaseEncodingStageTest):

  def default_encoding_stage(self):
    """See base class."""
    return test_utils.SignIntFloatEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.constant([0.0, 0.1, -0.1, 0.9, -0.9, 1.6, -2.2])

  @property
  def is_lossless(self):
    """See base class."""
    return True

  def common_asserts_for_test_data(self, data):
    """See base class."""
    signs, ints, floats = (
        data.encoded_x[test_utils.SignIntFloatEncodingStage.ENCODED_SIGNS_KEY],
        data.encoded_x[test_utils.SignIntFloatEncodingStage.ENCODED_INTS_KEY],
        data.encoded_x[test_utils.SignIntFloatEncodingStage.ENCODED_FLOATS_KEY])
    self.assertAllEqual(np.array([0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]), signs)
    self.assertAllEqual(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0]), ints)
    self.assertAllClose(np.array([0.0, 0.1, 0.1, 0.9, 0.9, 0.6, 0.2]), floats)


class PlusRandomNumEncodingStageTest(test_utils.BaseEncodingStageTest):

  _ENCODED_VALUES_KEY = test_utils.PlusRandomNumEncodingStage.ENCODED_VALUES_KEY

  def default_encoding_stage(self):
    """See base class."""
    return test_utils.PlusRandomNumEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.random.uniform([5])

  @property
  def is_lossless(self):
    """See base class."""
    return True

  def common_asserts_for_test_data(self, data):
    """See base class."""
    frac_x, _ = np.modf(data.x)
    frac_encoded_x, _ = np.modf(data.encoded_x[self._ENCODED_VALUES_KEY])
    # The decimal places should be the same.
    self.assertAllClose(
        frac_x,
        frac_encoded_x,
        rtol=test_utils.DEFAULT_RTOL,
        atol=test_utils.DEFAULT_ATOL)

  def test_encoding_differs_given_different_seed(self):
    """Tests that encoded_x is different in different evaluations."""
    x = tf.constant(self.evaluate(self.default_input()))
    stage = self.default_encoding_stage()
    encode_params, decode_params = stage.get_params()
    encoded_x, decoded_x = self.encode_decode_x(stage, x, encode_params,
                                                decode_params)
    test_data_1 = self.evaluate_test_data(
        test_utils.TestData(x, encoded_x, decoded_x))
    test_data_2 = self.evaluate_test_data(
        test_utils.TestData(x, encoded_x, decoded_x))

    # The decoded values should be the sam, but the encoded values not.
    self.assertAllClose(
        test_data_1.decoded_x,
        test_data_2.decoded_x,
        rtol=test_utils.DEFAULT_RTOL,
        atol=test_utils.DEFAULT_ATOL)
    self.assertNotAllClose(
        test_data_1.encoded_x[self._ENCODED_VALUES_KEY],
        test_data_2.encoded_x[self._ENCODED_VALUES_KEY],
        rtol=test_utils.DEFAULT_RTOL,
        atol=test_utils.DEFAULT_ATOL)


class PlusOneOverNEncodingStageTest(test_utils.BaseEncodingStageTest):

  _ENCODED_VALUES_KEY = test_utils.PlusOneOverNEncodingStage.ENCODED_VALUES_KEY
  _ADD_PARAM_KEY = test_utils.PlusOneOverNEncodingStage.ADD_PARAM_KEY
  _ITERATION_KEY = test_utils.PlusOneOverNEncodingStage.ITERATION_STATE_KEY

  def default_encoding_stage(self):
    """See base class."""
    return test_utils.PlusOneOverNEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.random.uniform([5])

  @property
  def is_lossless(self):
    """See base class."""
    return True

  def common_asserts_for_test_data(self, data):
    """See base class."""
    self.assertAllClose(data.x, data.decoded_x)
    self.assertAllClose(data.x + 1.0, data.encoded_x[self._ENCODED_VALUES_KEY])
    self.assertAllClose(data.initial_state[self._ITERATION_KEY] + 1.0,
                        data.updated_state[self._ITERATION_KEY])
    self.assertDictEqual(data.state_update_tensors, {})

  def test_one_to_many_few_rounds(self):
    """Encoding and decoding in the one-to-many setting for a few rounds.

    This is an example of how behavior of adaptive encoding stage can be tested
    over multiple iterations of encoding. The one-to-many setting does not
    include aggregation of state_update_tensors.
    """
    stage = self.default_encoding_stage()
    state = self.evaluate(stage.initial_state())
    for i in range(1, 5):
      data = self.run_one_to_many_encode_decode(stage, self.default_input,
                                                state)
      self.assertAllClose(data.x, data.decoded_x)
      self.assertAllClose(data.x + 1 / data.initial_state[self._ITERATION_KEY],
                          data.encoded_x[self._ENCODED_VALUES_KEY])
      self.assertEqual(data.initial_state[self._ITERATION_KEY], i)
      self.assertDictEqual(data.state_update_tensors, {})
      self.assertEqual(data.updated_state[self._ITERATION_KEY], i + 1.0)
      state = data.updated_state

  def test_many_to_one_few_rounds(self):
    """Encoding and decoding in the many-to-one setting for a few rounds.

    This is an example of how behavior of adaptive encoding stage can be tested
    over multiple iterations of encoding, including the aggregation of
    state_update_tensors.
    """
    stage = self.default_encoding_stage()
    state = self.evaluate(stage.initial_state())
    for i in range(1, 5):
      input_values = self.evaluate([self.default_input() for _ in range(3)])
      data, _ = self.run_many_to_one_encode_decode(stage, input_values, state)
      for d in data:
        self.assertAllClose(d.x, d.decoded_x)
        self.assertAllClose(d.x + 1 / d.initial_state[self._ITERATION_KEY],
                            d.encoded_x[self._ENCODED_VALUES_KEY])
        self.assertEqual(d.initial_state[self._ITERATION_KEY], i)
        self.assertDictEqual(d.state_update_tensors, {})
        self.assertEqual(d.updated_state[self._ITERATION_KEY], i + 1.0)
      state = data[0].updated_state


class AdaptiveNormalizeEncodingStageTest(test_utils.BaseEncodingStageTest):

  _VALUES_KEY = test_utils.AdaptiveNormalizeEncodingStage.ENCODED_VALUES_KEY
  _FACTOR_STATE_KEY = test_utils.AdaptiveNormalizeEncodingStage.FACTOR_STATE_KEY
  _NORM_STATE_UPDATE_KEY = (
      test_utils.AdaptiveNormalizeEncodingStage.NORM_STATE_UPDATE_KEY)

  def default_encoding_stage(self):
    """See base class."""
    return test_utils.AdaptiveNormalizeEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.random.normal([10])

  @property
  def is_lossless(self):
    """See base class."""
    return True

  def common_asserts_for_test_data(self, data):
    """See base class."""
    self.assertIn(self._NORM_STATE_UPDATE_KEY, data.state_update_tensors)

  def test_one_to_many_few_rounds(self):
    """Encoding and decoding in the one-to-many setting for a few rounds.

    This is an example of how behavior of adaptive encoding stage can be tested
    over multiple iterations of encoding. The one-to-many setting does not
    include aggregation of state_update_tensors.
    """
    stage = self.default_encoding_stage()
    state = self.evaluate(stage.initial_state())
    for _ in range(5):
      data = self.run_one_to_many_encode_decode(stage, lambda: tf.ones([10]),
                                                state)
      self.assertAllClose(data.x, data.decoded_x)
      # Make sure the estimated factor is decreasing.
      self.assertLess(data.updated_state[self._FACTOR_STATE_KEY],
                      data.initial_state[self._FACTOR_STATE_KEY])
      state = data.updated_state

    # Run a few more times and verify that the factor is close to correct value.
    # That means, the encoded values are close to having norm 1.
    for _ in range(30):
      data = self.run_one_to_many_encode_decode(stage, lambda: tf.ones([10]),
                                                state)
      state = data.updated_state
    encoded_norm = np.linalg.norm(data.encoded_x[self._VALUES_KEY])
    self.assertAllClose(1.0, encoded_norm, atol=0.1)

  def test_many_to_one_few_rounds(self):
    """Encoding and decoding in the many-to-one setting for a few rounds.

    This is an example of how behavior of adaptive encoding stage can be tested
    over multiple iterations of encoding, including the aggregation of
    state_update_tensors.
    """
    stage = self.default_encoding_stage()
    state = self.evaluate(stage.initial_state())
    input_values = list(np.ones((3, 10), dtype=np.float32))
    for _ in range(5):
      data, _ = self.run_many_to_one_encode_decode(stage, input_values, state)
      for d in data:
        self.assertAllClose(d.x, d.decoded_x)
        # Make sure the estimated factor is increasing.
        self.assertLess(d.updated_state[self._FACTOR_STATE_KEY],
                        d.initial_state[self._FACTOR_STATE_KEY])
      state = data[0].updated_state

    # Run a few more times and verify that the factor is close to correct value.
    # That means, the encoded values are close to having norm 1.
    for _ in range(20):
      # input_values = self.evaluate([self.default_input() for _ in range(3)])
      data, _ = self.run_many_to_one_encode_decode(stage, input_values, state)
      state = data[0].updated_state
    for d in data:
      encoded_norm = np.linalg.norm(d.encoded_x[self._VALUES_KEY])
      self.assertAllClose(1.0, encoded_norm, atol=0.002)


class StateUpdateTensorsEncodingStageTest(test_utils.BaseEncodingStageTest):

  def default_encoding_stage(self):
    """See base class."""
    return test_utils.StateUpdateTensorsEncodingStage()

  def default_input(self):
    """See base class."""
    return tf.random.uniform([5])

  @property
  def is_lossless(self):
    """See base class."""
    return True

  def common_asserts_for_test_data(self, data):
    """See base class."""
    self.assertAllClose(data.x, data.decoded_x)
    self.assertAllClose(
        data.x, data.encoded_x[
            test_utils.StateUpdateTensorsEncodingStage.ENCODED_VALUES_KEY])

  def test_state_aggregation(self):
    stage = self.default_encoding_stage()
    # If a new StateAggregationMode is added, this test class should be updated.
    self.assertSetEqual(
        set(stage.state_update_aggregation_modes.values()),
        set(encoding_stage.StateAggregationMode))

    state = self.evaluate(stage.initial_state())
    input_values = [
        np.array([1.0, 5.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-3.0, 12.0, 0.1])
    ]

    data, _ = self.run_many_to_one_encode_decode(stage, input_values, state)
    updated_state = data[0].updated_state
    self.assertEqual(
        15.1, updated_state[
            test_utils.StateUpdateTensorsEncodingStage.LAST_SUM_STATE_KEY])
    self.assertEqual(
        -3.0, updated_state[
            test_utils.StateUpdateTensorsEncodingStage.LAST_MIN_STATE_KEY])
    self.assertEqual(
        12.0, updated_state[
            test_utils.StateUpdateTensorsEncodingStage.LAST_MAX_STATE_KEY])
    self.assertEqual(
        9, updated_state[
            test_utils.StateUpdateTensorsEncodingStage.LAST_COUNT_STATE_KEY])

    data, _ = self.run_many_to_one_encode_decode(stage,
                                                 input_values + input_values,
                                                 state)
    updated_state = data[0].updated_state
    self.assertEqual(
        30.2, updated_state[
            test_utils.StateUpdateTensorsEncodingStage.LAST_SUM_STATE_KEY])
    self.assertEqual(
        -3.0, updated_state[
            test_utils.StateUpdateTensorsEncodingStage.LAST_MIN_STATE_KEY])
    self.assertEqual(
        12.0, updated_state[
            test_utils.StateUpdateTensorsEncodingStage.LAST_MAX_STATE_KEY])
    self.assertEqual(
        18, updated_state[
            test_utils.StateUpdateTensorsEncodingStage.LAST_COUNT_STATE_KEY])


class TestUtilsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for other utilities in `test_utils.py`."""

  def test_dummy_rng_source(self):
    default_seed = 1
    self.assertTrue(tf.is_tensor(test_utils.dummy_rng_source(default_seed, 1)))

    # Test that the outputs are different given different seeds.
    val_1 = self.evaluate(test_utils.dummy_rng_source(default_seed, 1))
    val_2 = self.evaluate(test_utils.dummy_rng_source(default_seed + 1, 1))
    self.assertNotEqual(val_1, val_2)

    # Test the output Tensor has the correct shape.
    self.assertEqual((3,), test_utils.dummy_rng_source(default_seed, 3).shape)
    self.assertEqual((5,), test_utils.dummy_rng_source(default_seed, 5).shape)

  def test_get_tensor_with_random_shape(self):
    x = test_utils.get_tensor_with_random_shape()
    self.assertIsInstance(x, tf.Tensor)
    self.assertFalse(x.shape.is_fully_defined())
    # Rank of the Tensor should be known, even though the dimension is not.
    self.assertEqual(1, x.shape.ndims)

    # Assert that unknown shape corresponds to a value of actually random shape
    # at execution time.
    samples = [self.evaluate(x) for _ in range(10)]
    self.assertGreater(len(set([len(s) for s in samples])), 1)

    # Test that source_fn has effect on the output values.
    x_uniform = test_utils.get_tensor_with_random_shape(
        expected_num_elements=50, source_fn=tf.random.uniform)
    x_normal = test_utils.get_tensor_with_random_shape(
        expected_num_elements=50, source_fn=tf.random.normal)
    self.assertGreaterEqual(self.evaluate(tf.reduce_min(x_uniform)), 0.0)
    self.assertLess(self.evaluate(tf.reduce_min(x_normal)), 0.0)

  def test_is_adaptive_stage(self):
    self.assertFalse(
        test_utils.is_adaptive_stage(test_utils.PlusOneEncodingStage()))
    self.assertTrue(
        test_utils.is_adaptive_stage(test_utils.PlusOneOverNEncodingStage()))

  @parameterized.parameters([1.0, 'str', object])
  def test_is_adaptive_stage_raises(self, not_a_stage):
    with self.assertRaises(TypeError):
      test_utils.is_adaptive_stage(not_a_stage)

  def test_aggregate_state_update_tensors(self):
    test_state_update_aggregation_modes = {
        'sum': encoding_stage.StateAggregationMode.SUM,
        'max': encoding_stage.StateAggregationMode.MAX,
        'min': encoding_stage.StateAggregationMode.MIN,
        'stack': encoding_stage.StateAggregationMode.STACK
    }
    # Ensure every option is captured by this test.
    self.assertSetEqual(
        set(encoding_stage.StateAggregationMode),
        set(test_state_update_aggregation_modes.values()))

    mock_stage = mock.Mock(
        spec=encoding_stage.AdaptiveEncodingStageInterface,
        state_update_aggregation_modes=test_state_update_aggregation_modes)
    array = np.array([[1.0, 2.0, -3.0], [-4.0, 5.0, 6.0]])
    state_update_tensors = [{
        'sum': array,
        'max': array,
        'min': array,
        'stack': array
    }, {
        'sum': -array,
        'max': -array,
        'min': -array,
        'stack': -array
    }]
    aggregated_tensors = test_utils.aggregate_state_update_tensors(
        mock_stage, state_update_tensors)

    self.assertAllEqual(aggregated_tensors['sum'], np.zeros((2, 3)))
    self.assertAllEqual(aggregated_tensors['max'],
                        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    self.assertAllEqual(aggregated_tensors['min'],
                        np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]]))
    self.assertAllEqual(
        aggregated_tensors['stack'],
        np.array([[[1.0, 2.0, -3.0], [-4.0, 5.0, 6.0]],
                  [[-1.0, -2.0, 3.0], [4.0, -5.0, -6.0]]]))


if __name__ == '__main__':
  tf.test.main()
