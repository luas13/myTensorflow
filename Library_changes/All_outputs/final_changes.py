import tensorflow as tf
import numpy as np

# Changing tensorflow codes

from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.util import nest
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope as vs

class myMultiRNNCell(RNNCell):
  """RNN cell composed sequentially of multiple simple cells."""

  def __init__(self, cells, state_is_tuple=True):
    """Create a RNN cell composed sequentially of a number of RNNCells.
    Args:
      cells: list of RNNCells that will be composed in this order.
      state_is_tuple: If True, accepted and returned states are n-tuples, where
        `n = len(cells)`.  If False, the states are all
        concatenated along the column axis.  This latter behavior will soon be
        deprecated.
    Raises:
      ValueError: if cells is empty (not allowed), or at least one of the cells
        returns a state tuple but the flag `state_is_tuple` is `False`.
    """
    if not cells:
      raise ValueError("Must specify at least one cell for MultiRNNCell.")
    if not nest.is_sequence(cells):
      raise TypeError(
          "cells must be a list or tuple, but saw: %s." % cells)

    self._cells = cells
    self._state_is_tuple = state_is_tuple
    if not state_is_tuple:
      if any(nest.is_sequence(c.state_size) for c in self._cells):
        raise ValueError("Some cells return tuples of states, but the flag "
                         "state_is_tuple is not set.  State sizes are: %s"
                         % str([c.state_size for c in self._cells]))

  @property
  def state_size(self):
    if self._state_is_tuple:
      return tuple(cell.state_size for cell in self._cells)
    else:
      return sum([cell.state_size for cell in self._cells])

  # Change the output_size to contain all layer's output
  @property
  def output_size(self):
    return tuple(cell.output_size for cell in self._cells)
    #return self._cells[-1].output_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      if self._state_is_tuple:
        return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)
      else:
        # We know here that state_size of each cell is not a tuple and
        # presumably does not contain TensorArrays or anything else fancy
        return super(MultiRNNCell, self).zero_state(batch_size, dtype)

  def __call__(self, inputs, state, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
    with vs.variable_scope(scope or "multi_rnn_cell"):
      cur_state_pos = 0
      cur_inp = inputs
      new_states = []
      # Create an output tuple to contain output of all layers
      new_outputs = []
      for i, cell in enumerate(self._cells):
        with vs.variable_scope("cell_%d" % i):
          if self._state_is_tuple:
            if not nest.is_sequence(state):
              raise ValueError(
                  "Expected state to be a tuple of length %d, but received: %s"
                  % (len(self.state_size), state))
            cur_state = state[i]
          else:
            cur_state = array_ops.slice(
                state, [0, cur_state_pos], [-1, cell.state_size])
            cur_state_pos += cell.state_size
          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)
          # Keep appending all the layer's output
          new_outputs.append(cur_inp)
    new_states = (tuple(new_states) if self._state_is_tuple else
                  array_ops.concat(new_states, 1))
    new_outputs = (tuple(new_outputs) if self._state_is_tuple else
                  array_ops.concat(new_outputs, 1))
    # Return the new_outputs in place of last layer's output
    return new_outputs, new_states
    #return cur_inp, new_states

# End of changes

sess = tf.InteractiveSession()

tf.reset_default_graph()

# Create input data
X = np.random.randn(2, 10, 8)

X_lengths = [10, 10]
num_layers = 3

cell = tf.contrib.rnn.LSTMCell(num_units=64, state_is_tuple=True)

#cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
cell = myMultiRNNCell([cell] * num_layers)

outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X)

result = tf.contrib.learn.run_n(
    {"outputs": outputs, "last_states": last_states},
    n=1,
    feed_dict=None)

print('\n')
print "results shape", len(result), "\n"

#print "result[0]['outputs'].shape", result[0]["outputs"].shape, "\n"

#print "For 1st layer result[0]['last_states'][0].c.shape", result[0]["last_states"][0].c.shape, "\n"
#print "For 1st layer result[0]['last_states'][0].h.shape", result[0]["last_states"][0].h.shape, "\n"
#
#print "For 2nd layer result[0]['last_states'][1].c.shape", result[0]["last_states"][1].c.shape, "\n"
#print "For 2nd layer result[0]['last_states'][1].h.shape", result[0]["last_states"][1].h.shape, "\n"
#
#print "For 3rd layer result[0]['last_states'][2].c.shape", result[0]["last_states"][2].c.shape, "\n"
#print "For 3rd layer result[0]['last_states'][2].h.shape", result[0]["last_states"][2].h.shape, "\n"

print "**************** Actual Values *********************\n"

print "Extracting ouputs via final_states returned by dynamic_rnn"

print "For 1st layer, across 2 batches, in last time-step, result[0]['last_states'][0].c \n", result[0]["last_states"][0].c, "\n"
print "For 1st layer, across 2 batches, in last time-step, result[0]['last_states'][0].h \n", result[0]["last_states"][0].h, "\n"

print "For 2nd layer, across 2 batches, in last time-step, result[0]['last_states'][1].c \n", result[0]["last_states"][1].c, "\n"
print "For 2nd layer, across 2 batches, in last time-step, result[0]['last_states'][1].h \n", result[0]["last_states"][1].h, "\n"

print "For 3rd layer, across 2 batches, in last time-step, result[0]['last_states'][2].c \n", result[0]["last_states"][2].c, "\n"
print "For 3rd layer, across 2 batches, in last time-step, result[0]['last_states'][2].h \n", result[0]["last_states"][2].h, "\n"

print "Extracting ouputs via outputs returned by dynamic_rnn", "\n"

print "For 1st layer, Total output for 1st batch across 10 time-steps via Outputs \n", (result[0]["outputs"][0][0]), "\n"
print "For 1st layer, Total output for 2nd batch across 10 time-steps via Outputs \n", (result[0]["outputs"][0][1]), "\n"

print "For 2nd layer, Total output for 1st batch across 10 time-steps via Outputs \n", (result[0]["outputs"][1][0]), "\n"
print "For 2nd layer, Total output for 2nd batch across 10 time-steps via Outputs \n", (result[0]["outputs"][1][1]), "\n"

print "For 3rd layer, Total output for 1st batch across 10 time-steps via Outputs \n", (result[0]["outputs"][2][0]), "\n"
print "For 3rd layer, Total output for 2nd batch across 10 time-steps via Outputs \n", (result[0]["outputs"][2][1]), "\n"

print "For 3rd layer, Total output for 2nd batch across 10th time-steps via Outputs \n", (result[0]["outputs"][2][1][9]), "\n"
