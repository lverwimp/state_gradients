#! /usr/bin/env python

import tensorflow as tf

class IntermStateWrapper(tf.contrib.rnn.RNNCell):
    '''
    This wraps an RNN cell to choose from which part
    of the LSTM cell we want the intermediate steps
    if to_return = cell, we get the intermediate cell states,
    otherwise the intermediate hidden states.
    Without this wrapper, the LSTM cell only returns the
    cell state from the last time step.
    '''

    def __init__(self, cell, to_return="cell"):
        '''IntermStateWrapper constructor'''

        self._cell = cell
        self._to_return = to_return

        if self._to_return != "cell" and self._to_return != "hidden":
            raise IOError("to_return should be 'cell' or 'hidden'")

    @property
    def output_size(self):
        '''return cell output size'''

        return self._cell.state_size[0]

    @property
    def state_size(self):
        '''return cell state size'''

        return self._cell.state_size

    def zero_state(self, batch_size, dtype):
        '''the cell zero state'''

        return self._cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, scope=None):
        '''call wrapped cell with constant scope'''

        # new_state = (c, h)
        _, new_state = self._cell(inputs, state, scope)

        if self._to_return == "cell":
            return new_state[0], new_state
        else:
            return new_state[1], new_state
