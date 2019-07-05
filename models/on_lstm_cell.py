from tensorflow.python.keras import activations
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

# for tensorflow 1.12
class OnLSTMCell(LayerRNNCell):
    def __init__(self,
                 num_units,
                 levels,
                 activation=None,
                 reuse=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(OnLSTMCell, self).__init__(
            _reuse=reuse, name=name, dtype=dtype, **kwargs
        )
        self.input_spec = input_spec.InputSpec(ndim=2)

        self._num_units = num_units
        self._levels = levels
        self.chunk_size = self._num_units / self._levels
        assert self._num_units % self._levels == 0
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
                             str(inputs_shape))
        input_depth = inputs_shape[-1]
        h_depth = self._num_units
        self._kernel = self.add_variable(
            "kernel",
            shape=[input_depth + h_depth, 4 *
                   self._num_units + 2 * self._levels],
            initializer=init_ops.glorot_uniform_initializer()
        )
        self._bias = self.add_variable(
            "bias",
            shape=[4 * self._num_units + 2 * self._levels],
            initializer=init_ops.zeros_initializer(dtype=self.dtype)
        )

        self.built = True

    @classmethod
    def cummax(cls, inputs, reverse=False):
        softmax = nn_ops.softmax(inputs, -1)
        return math_ops.cumsum(softmax, -1, reverse=reverse)

    def call(self, inputs, state):
        sigmoid = math_ops.sigmoid
        c, h = state
        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, h], 1), self._kernel
        )
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        master_f_gate = self.cummax(gate_inputs[:, :self._levels])
        master_f_gate = array_ops.expand_dims(master_f_gate, -1)
        master_i_gate = self.cummax(
            gate_inputs[:, self._levels:self._levels * 2], reversed=True)
        master_i_gate = array_ops.expand_dims(master_i_gate, -1)
        f, i, o, j = array_ops.split(
            value=gate_inputs[:, self._levels * 2:], num_or_size_splits=4, axis=None
        )
        c_last = array_ops.reshape(c, [-1, self.levels, self.chunk_size])
        overlap = master_f_gate * master_i_gate
        c_out = overlap * (sigmoid(f) * c_last + sigmoid(i) * c) + \
            (master_f_gate - overlap) * c_last + \
            (master_i_gate - overlap) * self._activation(j)
        h_out = sigmoid(o) * self._activation(c_out)
        new_c = array_ops.reshape(c_out, [-1, self._num_units])
        new_h = array_ops.reshape(h_out, [-1, self._num_units])

        new_state = LSTMStateTuple(new_c, new_h)
        return new_h, new_state
