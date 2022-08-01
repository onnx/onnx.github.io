
.. _l-onnx-doc-LSTM:

====
LSTM
====

.. contents::
    :local:


.. _l-onnx-op-lstm-14:

LSTM - 14
=========

**Version**

* **name**: `LSTM (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#LSTM>`_
* **domain**: **main**
* **since_version**: **14**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 14**.

**Summary**

Computes an one-layer LSTM. This operator is usually supported via some
custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`o` - output gate

`f` - forget gate

`c` - cell gate

`t` - time step (t-1 means previous time step)

`W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates

`R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates

`Wb[iofc]` - W bias vectors for input, output, forget, and cell gates

`Rb[iofc]` - R bias vectors for input, output, forget, and cell gates

`P[iof]`  - P peephole weight vector for input, output, and forget gates

`WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates

`RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates

`WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates

`RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates

`PB[iof]`  - P peephole weight vector for backward input, output, and forget gates

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

  - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)

  - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)

  - ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)

  - Ct = ft (.) Ct-1 + it (.) ct

  - ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)

  - Ht = ot (.) h(Ct)
This operator has **optional** inputs/outputs. See `ONNX <https://github.com/onnx/onnx/blob/master/docs/IR.md>`_ for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

**Attributes**

* **activation_alpha**:
  Optional scaling values used by some activation functions. The
  values are consumed in the order of activation functions, for
  example (f, g, h) in LSTM. Default values are the same as of
  corresponding ONNX operators.For example with LeakyRelu, the default
  alpha is 0.01.
* **activation_beta**:
  Optional scaling values used by some activation functions. The
  values are consumed in the order of activation functions, for
  example (f, g, h) in LSTM. Default values are the same as of
  corresponding ONNX operators.
* **activations**:
  A list of 3 (or 6 if bidirectional) activation functions for input,
  output, forget, cell, and hidden. The activation functions must be
  one of the activation functions specified above. Optional: See the
  equations for default if not specified.
* **clip**:
  Cell clip threshold. Clipping bounds the elements of a tensor in the
  range of [-threshold, +threshold] and is applied to the input of
  activations. No clip if not specified.
* **direction**:
  Specify if the RNN is forward, reverse, or bidirectional. Must be
  one of forward (default), reverse, or bidirectional. Default value is ``'forward'``.
* **hidden_size**:
  Number of neurons in the hidden layer
* **input_forget**:
  Couple the input and forget gates if 1. Default value is ``0``.
* **layout**:
  The shape format of inputs X, initial_h, initial_c and outputs Y,
  Y_h, Y_c. If 0, the following shapes are expected: X.shape =
  [seq_length, batch_size, input_size], Y.shape = [seq_length,
  num_directions, batch_size, hidden_size], initial_h.shape =
  Y_h.shape = initial_c.shape = Y_c.shape = [num_directions,
  batch_size, hidden_size]. If 1, the following shapes are expected:
  X.shape = [batch_size, seq_length, input_size], Y.shape =
  [batch_size, seq_length, num_directions, hidden_size],
  initial_h.shape = Y_h.shape = initial_c.shape = Y_c.shape =
  [batch_size, num_directions, hidden_size]. Default value is ``0``.

**Inputs**

Between 3 and 8 inputs.

* **X** (heterogeneous) - **T**:
  The input sequences packed (and potentially padded) into one 3-D
  tensor with the shape of `[seq_length, batch_size, input_size]`.
* **W** (heterogeneous) - **T**:
  The weight tensor for the gates. Concatenation of `W[iofc]` and
  `WB[iofc]` (if bidirectional) along dimension 0. The tensor has
  shape `[num_directions, 4*hidden_size, input_size]`.
* **R** (heterogeneous) - **T**:
  The recurrence weight tensor. Concatenation of `R[iofc]` and
  `RB[iofc]` (if bidirectional) along dimension 0. This tensor has
  shape `[num_directions, 4*hidden_size, hidden_size]`.
* **B** (optional, heterogeneous) - **T**:
  The bias tensor for input gate. Concatenation of `[Wb[iofc],
  Rb[iofc]]`, and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along
  dimension 0. This tensor has shape `[num_directions,
  8*hidden_size]`. Optional: If not specified - assumed to be 0.
* **sequence_lens** (optional, heterogeneous) - **T1**:
  Optional tensor specifying lengths of the sequences in a batch. If
  not specified - assumed all sequences in the batch to have length
  `seq_length`. It has shape `[batch_size]`.
* **initial_h** (optional, heterogeneous) - **T**:
  Optional initial value of the hidden. If not specified - assumed to
  be 0. It has shape `[num_directions, batch_size, hidden_size]`.
* **initial_c** (optional, heterogeneous) - **T**:
  Optional initial value of the cell. If not specified - assumed to be
  0. It has shape `[num_directions, batch_size, hidden_size]`.
* **P** (optional, heterogeneous) - **T**:
  The weight tensor for peepholes. Concatenation of `P[iof]` and
  `PB[iof]` (if bidirectional) along dimension 0. It has shape
  `[num_directions, 3*hidde_size]`. Optional: If not specified -
  assumed to be 0.

**Outputs**

Between 0 and 3 outputs.

* **Y** (optional, heterogeneous) - **T**:
  A tensor that concats all the intermediate output values of the
  hidden. It has shape `[seq_length, num_directions, batch_size,
  hidden_size]`.
* **Y_h** (optional, heterogeneous) - **T**:
  The last output value of the hidden. It has shape `[num_directions,
  batch_size, hidden_size]`.
* **Y_c** (optional, heterogeneous) - **T**:
  The last output value of the cell. It has shape `[num_directions,
  batch_size, hidden_size]`.

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
* **T1** in (
  tensor(int32)
  ):
  Constrain seq_lens to integer tensor.

**Examples**

**defaults**

::

    input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

    input_size = 2
    hidden_size = 3
    weight_scale = 0.1
    number_of_gates = 4

    node = onnx.helper.make_node(
        'LSTM',
        inputs=['X', 'W', 'R'],
        outputs=['', 'Y_h'],
        hidden_size=hidden_size
    )

    W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
    R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

    lstm = LSTM_Helper(X=input, W=W, R=R)
    _, Y_h = lstm.step()
    expect(node, inputs=[input, W, R], outputs=[Y_h.astype(np.float32)], name='test_lstm_defaults')

**initial_bias**

::

    input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]).astype(np.float32)

    input_size = 3
    hidden_size = 4
    weight_scale = 0.1
    custom_bias = 0.1
    number_of_gates = 4

    node = onnx.helper.make_node(
        'LSTM',
        inputs=['X', 'W', 'R', 'B'],
        outputs=['', 'Y_h'],
        hidden_size=hidden_size
    )

    W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
    R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

    # Adding custom bias
    W_B = custom_bias * np.ones((1, number_of_gates * hidden_size)).astype(np.float32)
    R_B = np.zeros((1, number_of_gates * hidden_size)).astype(np.float32)
    B = np.concatenate((W_B, R_B), 1)

    lstm = LSTM_Helper(X=input, W=W, R=R, B=B)
    _, Y_h = lstm.step()
    expect(node, inputs=[input, W, R, B], outputs=[Y_h.astype(np.float32)], name='test_lstm_with_initial_bias')

**peepholes**

::

    input = np.array([[[1., 2., 3., 4.], [5., 6., 7., 8.]]]).astype(np.float32)

    input_size = 4
    hidden_size = 3
    weight_scale = 0.1
    number_of_gates = 4
    number_of_peepholes = 3

    node = onnx.helper.make_node(
        'LSTM',
        inputs=['X', 'W', 'R', 'B', 'sequence_lens', 'initial_h', 'initial_c', 'P'],
        outputs=['', 'Y_h'],
        hidden_size=hidden_size
    )

    # Initializing Inputs
    W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
    R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)
    B = np.zeros((1, 2 * number_of_gates * hidden_size)).astype(np.float32)
    seq_lens = np.repeat(input.shape[0], input.shape[1]).astype(np.int32)
    init_h = np.zeros((1, input.shape[1], hidden_size)).astype(np.float32)
    init_c = np.zeros((1, input.shape[1], hidden_size)).astype(np.float32)
    P = weight_scale * np.ones((1, number_of_peepholes * hidden_size)).astype(np.float32)

    lstm = LSTM_Helper(X=input, W=W, R=R, B=B, P=P, initial_c=init_c, initial_h=init_h)
    _, Y_h = lstm.step()
    expect(node, inputs=[input, W, R, B, seq_lens, init_h, init_c, P], outputs=[Y_h.astype(np.float32)],
           name='test_lstm_with_peepholes')

**batchwise**

::

    input = np.array([[[1., 2.]], [[3., 4.]], [[5., 6.]]]).astype(np.float32)

    input_size = 2
    hidden_size = 7
    weight_scale = 0.3
    number_of_gates = 4
    layout = 1

    node = onnx.helper.make_node(
        'LSTM',
        inputs=['X', 'W', 'R'],
        outputs=['Y', 'Y_h'],
        hidden_size=hidden_size,
        layout=layout
    )

    W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
    R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

    lstm = LSTM_Helper(X=input, W=W, R=R, layout=layout)
    Y, Y_h = lstm.step()
    expect(node, inputs=[input, W, R], outputs=[Y.astype(np.float32), Y_h.astype(np.float32)], name='test_lstm_batchwise')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Computes an one-layer LSTM. This operator is usually supported via some</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Computes an one-layer LSTM. This operator is usually supported via some</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">custom implementation such as CuDNN.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">custom implementation such as CuDNN.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Notations:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Notations:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">X - input tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">X - input tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">i - input gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">i - input gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">o - output gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">o - output gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">f - forget gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">f - forget gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">c - cell gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">c - cell gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">t - time step (t-1 means previous time step)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">t - time step (t-1 means previous time step)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">W[iofc] - W parameter weight matrix for input, output, forget, and cell gates</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">W[iofc] - W parameter weight matrix for input, output, forget, and cell gates</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">R[iofc] - R recurrence weight matrix for input, output, forget, and cell gates</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">R[iofc] - R recurrence weight matrix for input, output, forget, and cell gates</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Wb[iofc] - W bias vectors for input, output, forget, and cell gates</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Wb[iofc] - W bias vectors for input, output, forget, and cell gates</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Rb[iofc] - R bias vectors for input, output, forget, and cell gates</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Rb[iofc] - R bias vectors for input, output, forget, and cell gates</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">P[iof]  - P peephole weight vector for input, output, and forget gates</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">P[iof]  - P peephole weight vector for input, output, and forget gates</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">WB[iofc] - W parameter weight matrix for backward input, output, forget, and cell gates</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">WB[iofc] - W parameter weight matrix for backward input, output, forget, and cell gates</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">RB[iofc] - R recurrence weight matrix for backward input, output, forget, and cell gates</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">RB[iofc] - R recurrence weight matrix for backward input, output, forget, and cell gates</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">WBb[iofc] - W bias vectors for backward input, output, forget, and cell gates</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">WBb[iofc] - W bias vectors for backward input, output, forget, and cell gates</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">RBb[iofc] - R bias vectors for backward input, output, forget, and cell gates</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">RBb[iofc] - R bias vectors for backward input, output, forget, and cell gates</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">PB[iof]  - P peephole weight vector for backward input, output, and forget gates</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">PB[iof]  - P peephole weight vector for backward input, output, and forget gates</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">H - Hidden state</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">H - Hidden state</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">num_directions - 2 if direction == bidirectional else 1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">num_directions - 2 if direction == bidirectional else 1</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Activation functions:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Activation functions:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Relu(x)                - max(0, x)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Relu(x)                - max(0, x)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Sigmoid(x)             - 1/(1 + e^{-x})</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Sigmoid(x)             - 1/(1 + e^{-x})</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (NOTE: Below are optional)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (NOTE: Below are optional)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Affine(x)              - alpha*x + beta</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Affine(x)              - alpha*x + beta</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  LeakyRelu(x)           - x if x >= 0 else alpha * x</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  LeakyRelu(x)           - x if x >= 0 else alpha * x</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ThresholdedRelu(x)     - x if x >= alpha else 0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ThresholdedRelu(x)     - x if x >= alpha else 0</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ScaledTanh(x)          - alpha*Tanh(beta*x)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ScaledTanh(x)          - alpha*Tanh(beta*x)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Softsign(x)            - x/(1 + |x|)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Softsign(x)            - x/(1 + |x|)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Softplus(x)            - log(1 + e^x)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Softplus(x)            - log(1 + e^x)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  - ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  - ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  - Ct = ft (.) Ct-1 + it (.) ct</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  - Ct = ft (.) Ct-1 + it (.) ct</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  - ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  - ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  - Ht = ot (.) h(Ct)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  - Ht = ot (.) h(Ct)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator has **optional** inputs/outputs. See ONNX <https://github.com/onnx/onnx/blob/master/docs/IR.md>_ for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator has **optional** inputs/outputs. See ONNX <https://github.com/onnx/onnx/blob/master/docs/IR.md>_ for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activation_alpha**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activation_alpha**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional scaling values used by some activation functions. The</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional scaling values used by some activation functions. The</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values are consumed in the order of activation functions, for</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values are consumed in the order of activation functions, for</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  example (f, g, h) in LSTM. Default values are the same as of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  example (f, g, h) in LSTM. Default values are the same as of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding ONNX operators.For example with LeakyRelu, the default</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding ONNX operators.For example with LeakyRelu, the default</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  alpha is 0.01.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  alpha is 0.01.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activation_beta**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activation_beta**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional scaling values used by some activation functions. The</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional scaling values used by some activation functions. The</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values are consumed in the order of activation functions, for</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values are consumed in the order of activation functions, for</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  example (f, g, h) in LSTM. Default values are the same as of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  example (f, g, h) in LSTM. Default values are the same as of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding ONNX operators.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding ONNX operators.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activations**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activations**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A list of 3 (or 6 if bidirectional) activation functions for input,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A list of 3 (or 6 if bidirectional) activation functions for input,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output, forget, cell, and hidden. The activation functions must be</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output, forget, cell, and hidden. The activation functions must be</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  one of the activation functions specified above. Optional: See the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  one of the activation functions specified above. Optional: See the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  equations for default if not specified.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  equations for default if not specified.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **clip**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **clip**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Cell clip threshold. Clipping bounds the elements of a tensor in the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Cell clip threshold. Clipping bounds the elements of a tensor in the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  range of [-threshold, +threshold] and is applied to the input of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  range of [-threshold, +threshold] and is applied to the input of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  activations. No clip if not specified.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  activations. No clip if not specified.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **direction**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **direction**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Specify if the RNN is forward, reverse, or bidirectional. Must be</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Specify if the RNN is forward, reverse, or bidirectional. Must be</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  one of forward (default), reverse, or bidirectional. Default value is 'forward'.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  one of forward (default), reverse, or bidirectional. Default value is 'forward'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **hidden_size**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **hidden_size**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Number of neurons in the hidden layer</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Number of neurons in the hidden layer</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">109</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">109</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **input_forget**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **input_forget**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">110</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">110</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Couple the input and forget gates if 1. Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Couple the input and forget gates if 1. Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">111</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **layout**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">112</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  The shape format of inputs X, initial_h, initial_c and outputs Y,</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">113</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Y_h, Y_c. If 0, the following shapes are expected: X.shape =</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">114</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  [seq_length, batch_size, input_size], Y.shape = [seq_length,</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">115</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  num_directions, batch_size, hidden_size], initial_h.shape =</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">116</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Y_h.shape = initial_c.shape = Y_c.shape = [num_directions,</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">117</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  batch_size, hidden_size]. If 1, the following shapes are expected:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">118</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  X.shape = [batch_size, seq_length, input_size], Y.shape =</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">119</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  [batch_size, seq_length, num_directions, hidden_size],</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">120</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  initial_h.shape = Y_h.shape = initial_c.shape = Y_c.shape =</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">121</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  [batch_size, num_directions, hidden_size]. Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">111</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">122</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">112</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">123</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">113</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">124</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">114</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">125</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 3 and 8 inputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 3 and 8 inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">115</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">126</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">116</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">127</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">117</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">128</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input sequences packed (and potentially padded) into one 3-D</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input sequences packed (and potentially padded) into one 3-D</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">118</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">129</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor with the shape of [seq_length, batch_size, input_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor with the shape of [seq_length, batch_size, input_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">119</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">130</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **W** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **W** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">120</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">131</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The weight tensor for the gates. Concatenation of W[iofc] and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The weight tensor for the gates. Concatenation of W[iofc] and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">121</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">132</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  WB[iofc] (if bidirectional) along dimension 0. The tensor has</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  WB[iofc] (if bidirectional) along dimension 0. The tensor has</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">122</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">133</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  shape [num_directions, 4*hidden_size, input_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  shape [num_directions, 4*hidden_size, input_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">123</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">134</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **R** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **R** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">124</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">135</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The recurrence weight tensor. Concatenation of R[iofc] and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The recurrence weight tensor. Concatenation of R[iofc] and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">125</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">136</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  RB[iofc] (if bidirectional) along dimension 0. This tensor has</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  RB[iofc] (if bidirectional) along dimension 0. This tensor has</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">126</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">137</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  shape [num_directions, 4*hidden_size, hidden_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  shape [num_directions, 4*hidden_size, hidden_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">127</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">138</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">128</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">139</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The bias tensor for input gate. Concatenation of [Wb[iofc],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The bias tensor for input gate. Concatenation of [Wb[iofc],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">129</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">140</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Rb[iofc]], and [WBb[iofc], RBb[iofc]] (if bidirectional) along</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Rb[iofc]], and [WBb[iofc], RBb[iofc]] (if bidirectional) along</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">130</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">141</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimension 0. This tensor has shape [num_directions,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimension 0. This tensor has shape [num_directions,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">131</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">142</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  8*hidden_size]. Optional: If not specified - assumed to be 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  8*hidden_size]. Optional: If not specified - assumed to be 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">132</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">143</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **sequence_lens** (optional, heterogeneous) - **T1**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **sequence_lens** (optional, heterogeneous) - **T1**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">133</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">144</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional tensor specifying lengths of the sequences in a batch. If</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional tensor specifying lengths of the sequences in a batch. If</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">134</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">145</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  not specified - assumed all sequences in the batch to have length</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  not specified - assumed all sequences in the batch to have length</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">135</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">146</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq_length. It has shape [batch_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq_length. It has shape [batch_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">136</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">147</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **initial_h** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **initial_h** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">137</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">148</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional initial value of the hidden. If not specified - assumed to</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional initial value of the hidden. If not specified - assumed to</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">138</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">149</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  be 0. It has shape [num_directions, batch_size, hidden_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  be 0. It has shape [num_directions, batch_size, hidden_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">139</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">150</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **initial_c** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **initial_c** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">140</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">151</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional initial value of the cell. If not specified - assumed to be</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional initial value of the cell. If not specified - assumed to be</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">141</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">152</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  0. It has shape [num_directions, batch_size, hidden_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  0. It has shape [num_directions, batch_size, hidden_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">142</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">153</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **P** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **P** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">143</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">154</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The weight tensor for peepholes. Concatenation of P[iof] and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The weight tensor for peepholes. Concatenation of P[iof] and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">144</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">155</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  PB[iof] (if bidirectional) along dimension 0. It has shape</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  PB[iof] (if bidirectional) along dimension 0. It has shape</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">145</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">156</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  [num_directions, 3*hidde_size]. Optional: If not specified -</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  [num_directions, 3*hidde_size]. Optional: If not specified -</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">146</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">157</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  assumed to be 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  assumed to be 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">147</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">158</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">148</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">159</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">149</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">160</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">150</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">161</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 0 and 3 outputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 0 and 3 outputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">151</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">162</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">152</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">163</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">153</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">164</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A tensor that concats all the intermediate output values of the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A tensor that concats all the intermediate output values of the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">154</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">165</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  hidden. It has shape [seq_length, num_directions, batch_size,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  hidden. It has shape [seq_length, num_directions, batch_size,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">155</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">166</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  hidden_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  hidden_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">156</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">167</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y_h** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y_h** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">157</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">168</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The last output value of the hidden. It has shape [num_directions,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The last output value of the hidden. It has shape [num_directions,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">158</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">169</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_size, hidden_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_size, hidden_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">159</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">170</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y_c** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y_c** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">160</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">171</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The last output value of the cell. It has shape [num_directions,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The last output value of the cell. It has shape [num_directions,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">161</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">172</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_size, hidden_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_size, hidden_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">162</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">173</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">163</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">174</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">164</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">175</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">165</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">176</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">166</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">177</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">167</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">178</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">168</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">179</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">169</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">180</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">170</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">181</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">171</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">182</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">172</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">183</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">173</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">184</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">174</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">185</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain seq_lens to integer tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain seq_lens to integer tensor.</code></td></tr>
    </table>

.. _l-onnx-op-lstm-7:

LSTM - 7
========

**Version**

* **name**: `LSTM (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#LSTM>`_
* **domain**: **main**
* **since_version**: **7**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 7**.

**Summary**

Computes an one-layer LSTM. This operator is usually supported via some
custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`o` - output gate

`f` - forget gate

`c` - cell gate

`t` - time step (t-1 means previous time step)

`W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates

`R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates

`Wb[iofc]` - W bias vectors for input, output, forget, and cell gates

`Rb[iofc]` - R bias vectors for input, output, forget, and cell gates

`P[iof]`  - P peephole weight vector for input, output, and forget gates

`WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates

`RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates

`WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates

`RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates

`PB[iof]`  - P peephole weight vector for backward input, output, and forget gates

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

  - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)

  - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)

  - ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)

  - Ct = ft (.) Ct-1 + it (.) ct

  - ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)

  - Ht = ot (.) h(Ct)
This operator has **optional** inputs/outputs. See `ONNX <https://github.com/onnx/onnx/blob/master/docs/IR.md>`_ for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

**Attributes**

* **activation_alpha**:
  Optional scaling values used by some activation functions. The
  values are consumed in the order of activation functions, for
  example (f, g, h) in LSTM. Default values are the same as of
  corresponding ONNX operators.For example with LeakyRelu, the default
  alpha is 0.01.
* **activation_beta**:
  Optional scaling values used by some activation functions. The
  values are consumed in the order of activation functions, for
  example (f, g, h) in LSTM. Default values are the same as of
  corresponding ONNX operators.
* **activations**:
  A list of 3 (or 6 if bidirectional) activation functions for input,
  output, forget, cell, and hidden. The activation functions must be
  one of the activation functions specified above. Optional: See the
  equations for default if not specified.
* **clip**:
  Cell clip threshold. Clipping bounds the elements of a tensor in the
  range of [-threshold, +threshold] and is applied to the input of
  activations. No clip if not specified.
* **direction**:
  Specify if the RNN is forward, reverse, or bidirectional. Must be
  one of forward (default), reverse, or bidirectional. Default value is ``'forward'``.
* **hidden_size**:
  Number of neurons in the hidden layer
* **input_forget**:
  Couple the input and forget gates if 1. Default value is ``0``.

**Inputs**

Between 3 and 8 inputs.

* **X** (heterogeneous) - **T**:
  The input sequences packed (and potentially padded) into one 3-D
  tensor with the shape of `[seq_length, batch_size, input_size]`.
* **W** (heterogeneous) - **T**:
  The weight tensor for the gates. Concatenation of `W[iofc]` and
  `WB[iofc]` (if bidirectional) along dimension 0. The tensor has
  shape `[num_directions, 4*hidden_size, input_size]`.
* **R** (heterogeneous) - **T**:
  The recurrence weight tensor. Concatenation of `R[iofc]` and
  `RB[iofc]` (if bidirectional) along dimension 0. This tensor has
  shape `[num_directions, 4*hidden_size, hidden_size]`.
* **B** (optional, heterogeneous) - **T**:
  The bias tensor for input gate. Concatenation of `[Wb[iofc],
  Rb[iofc]]`, and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along
  dimension 0. This tensor has shape `[num_directions,
  8*hidden_size]`. Optional: If not specified - assumed to be 0.
* **sequence_lens** (optional, heterogeneous) - **T1**:
  Optional tensor specifying lengths of the sequences in a batch. If
  not specified - assumed all sequences in the batch to have length
  `seq_length`. It has shape `[batch_size]`.
* **initial_h** (optional, heterogeneous) - **T**:
  Optional initial value of the hidden. If not specified - assumed to
  be 0. It has shape `[num_directions, batch_size, hidden_size]`.
* **initial_c** (optional, heterogeneous) - **T**:
  Optional initial value of the cell. If not specified - assumed to be
  0. It has shape `[num_directions, batch_size, hidden_size]`.
* **P** (optional, heterogeneous) - **T**:
  The weight tensor for peepholes. Concatenation of `P[iof]` and
  `PB[iof]` (if bidirectional) along dimension 0. It has shape
  `[num_directions, 3*hidde_size]`. Optional: If not specified -
  assumed to be 0.

**Outputs**

Between 0 and 3 outputs.

* **Y** (optional, heterogeneous) - **T**:
  A tensor that concats all the intermediate output values of the
  hidden. It has shape `[seq_length, num_directions, batch_size,
  hidden_size]`.
* **Y_h** (optional, heterogeneous) - **T**:
  The last output value of the hidden. It has shape `[num_directions,
  batch_size, hidden_size]`.
* **Y_c** (optional, heterogeneous) - **T**:
  The last output value of the cell. It has shape `[num_directions,
  batch_size, hidden_size]`.

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
* **T1** in (
  tensor(int32)
  ):
  Constrain seq_lens to integer tensor.

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Computes an one-layer LSTM. This operator is usually supported via some</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Computes an one-layer LSTM. This operator is usually supported via some</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">custom implementation such as CuDNN.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">custom implementation such as CuDNN.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Notations:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Notations:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">X - input tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">X - input tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">i - input gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">i - input gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">o - output gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">o - output gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">f - forget gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">f - forget gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">c - cell gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">c - cell gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">t - time step (t-1 means previous time step)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">t - time step (t-1 means previous time step)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">W[iofc] - W parameter weight matrix for input, output, forget, and cell gates</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">W[iofc] - W parameter weight matrix for input, output, forget, and cell gates</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">R[iofc] - R recurrence weight matrix for input, output, forget, and cell gates</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">R[iofc] - R recurrence weight matrix for input, output, forget, and cell gates</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Wb[iofc] - W bias vectors for input, output, forget, and cell gates</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Wb[iofc] - W bias vectors for input, output, forget, and cell gates</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Rb[iofc] - R bias vectors for input, output, forget, and cell gates</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Rb[iofc] - R bias vectors for input, output, forget, and cell gates</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">P[iof]  - P peephole weight vector for input, output, and forget gates</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">P[iof]  - P peephole weight vector for input, output, and forget gates</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">WB[iofc] - W parameter weight matrix for backward input, output, forget, and cell gates</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">WB[iofc] - W parameter weight matrix for backward input, output, forget, and cell gates</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">RB[iofc] - R recurrence weight matrix for backward input, output, forget, and cell gates</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">RB[iofc] - R recurrence weight matrix for backward input, output, forget, and cell gates</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">WBb[iofc] - W bias vectors for backward input, output, forget, and cell gates</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">WBb[iofc] - W bias vectors for backward input, output, forget, and cell gates</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">RBb[iofc] - R bias vectors for backward input, output, forget, and cell gates</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">RBb[iofc] - R bias vectors for backward input, output, forget, and cell gates</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">PB[iof]  - P peephole weight vector for backward input, output, and forget gates</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">PB[iof]  - P peephole weight vector for backward input, output, and forget gates</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">H - Hidden state</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">H - Hidden state</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">num_directions - 2 if direction == bidirectional else 1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">num_directions - 2 if direction == bidirectional else 1</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Activation functions:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Activation functions:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Relu(x)                - max(0, x)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Relu(x)                - max(0, x)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Sigmoid(x)             - 1/(1 + e^{-x})</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Sigmoid(x)             - 1/(1 + e^{-x})</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (NOTE: Below are optional)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (NOTE: Below are optional)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Affine(x)              - alpha*x + beta</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Affine(x)              - alpha*x + beta</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  LeakyRelu(x)           - x if x >= 0 else alpha * x</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  LeakyRelu(x)           - x if x >= 0 else alpha * x</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ThresholdedRelu(x)     - x if x >= alpha else 0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ThresholdedRelu(x)     - x if x >= alpha else 0</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ScaledTanh(x)          - alpha*Tanh(beta*x)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ScaledTanh(x)          - alpha*Tanh(beta*x)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Softsign(x)            - x/(1 + |x|)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Softsign(x)            - x/(1 + |x|)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Softplus(x)            - log(1 + e^x)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Softplus(x)            - log(1 + e^x)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td><code>69</code></td><td><code>69</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  - it = f(Xt*(Wi^T) + Ht-1*Ri + Pi (.) Ct-1 + Wbi + Rbi)</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  - it = f(Xt*(Wi^T) + Ht-1*<span style="color:#196F3D;">(</span>Ri<span style="color:#196F3D;">^</span><span style="color:#196F3D;">T</span><span style="color:#196F3D;">)</span> + Pi (.) Ct-1 + Wbi + Rbi)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td><code>71</code></td><td><code>71</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  - ft = f(Xt*(Wf^T) + Ht-1*Rf + Pf (.) Ct-1 + Wbf + Rbf)</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  - ft = f(Xt*(Wf^T) + Ht-1*<span style="color:#196F3D;">(</span>Rf<span style="color:#196F3D;">^</span><span style="color:#196F3D;">T</span><span style="color:#196F3D;">)</span> + Pf (.) Ct-1 + Wbf + Rbf)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td><code>73</code></td><td><code>73</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  - ct = g(Xt*(Wc^T) + Ht-1*Rc + Wbc + Rbc)</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  - ct = g(Xt*(Wc^T) + Ht-1*<span style="color:#196F3D;">(</span>Rc<span style="color:#196F3D;">^</span><span style="color:#196F3D;">T</span><span style="color:#196F3D;">)</span> + Wbc + Rbc)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  - Ct = ft (.) Ct-1 + it (.) ct</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  - Ct = ft (.) Ct-1 + it (.) ct</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td><code>77</code></td><td><code>77</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  - ot = f(Xt*(Wo^T) + Ht-1*Ro + Po (.) Ct + Wbo + Rbo)</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  - ot = f(Xt*(Wo^T) + Ht-1*<span style="color:#196F3D;">(</span>Ro<span style="color:#196F3D;">^</span><span style="color:#196F3D;">T</span><span style="color:#196F3D;">)</span> + Po (.) Ct + Wbo + Rbo)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">79</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  - Ht = ot (.) h(Ct)</code></td></tr>
    <tr style="1px solid black;"><td><code>79</code></td><td><code>80</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  <span style="color:#BA4A00;">-</span> <span style="color:#BA4A00;">H</span>t <span style="color:#BA4A00;">=</span> ot <span style="color:#BA4A00;">(</span>.<span style="color:#BA4A00;">)</span> h(<span style="color:#BA4A00;">C</span>t)</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">T</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span> <span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span> <span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">s</span> <span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span>t<span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span> <span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">.</span> <span style="color:#196F3D;">S</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">O</span><span style="color:#196F3D;">N</span><span style="color:#196F3D;">N</span><span style="color:#196F3D;">X</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"><</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">:</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">c</span>o<span style="color:#196F3D;">m</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">s</span>t<span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">I</span><span style="color:#196F3D;">R</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">></span><span style="color:#196F3D;">_</span> <span style="color:#196F3D;">f</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">s</span>. <span style="color:#196F3D;">A</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span>h<span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">'</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">T</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span>(t<span style="color:#196F3D;">h</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">w</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span>)<span style="color:#196F3D;"> </span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activation_alpha**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activation_alpha**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional scaling values used by some activation functions. The</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional scaling values used by some activation functions. The</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values are consumed in the order of activation functions, for</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values are consumed in the order of activation functions, for</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  example (f, g, h) in LSTM. Default values are the same as of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  example (f, g, h) in LSTM. Default values are the same as of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding ONNX operators.For example with LeakyRelu, the default</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding ONNX operators.For example with LeakyRelu, the default</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  alpha is 0.01.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  alpha is 0.01.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activation_beta**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activation_beta**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional scaling values used by some activation functions. The</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional scaling values used by some activation functions. The</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values are consumed in the order of activation functions, for</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values are consumed in the order of activation functions, for</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  example (f, g, h) in LSTM. Default values are the same as of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  example (f, g, h) in LSTM. Default values are the same as of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding ONNX operators.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding ONNX operators.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activations**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activations**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A list of 3 (or 6 if bidirectional) activation functions for input,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A list of 3 (or 6 if bidirectional) activation functions for input,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output, forget, cell, and hidden. The activation functions must be</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output, forget, cell, and hidden. The activation functions must be</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  one of the activation functions specified above. Optional: See the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  one of the activation functions specified above. Optional: See the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  equations for default if not specified.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  equations for default if not specified.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **clip**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **clip**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Cell clip threshold. Clipping bounds the elements of a tensor in the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Cell clip threshold. Clipping bounds the elements of a tensor in the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  range of [-threshold, +threshold] and is applied to the input of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  range of [-threshold, +threshold] and is applied to the input of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  activations. No clip if not specified.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  activations. No clip if not specified.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **direction**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **direction**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Specify if the RNN is forward, reverse, or bidirectional. Must be</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Specify if the RNN is forward, reverse, or bidirectional. Must be</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  one of forward (default), reverse, or bidirectional. Default value is 'forward'.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  one of forward (default), reverse, or bidirectional. Default value is 'forward'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **hidden_size**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **hidden_size**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Number of neurons in the hidden layer</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Number of neurons in the hidden layer</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">109</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **input_forget**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **input_forget**:</code></td></tr>
    <tr style="1px solid black;"><td><code>109</code></td><td><code>110</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Couple the input and forget gates if 1<span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">0</span>. Default value is 0.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Couple the input and forget gates if 1. Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">110</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">* **output_sequence**:</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">111</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  The sequence output for the hidden is optional if 0. Default 0. Default value is 0.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">112</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">111</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">113</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">112</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">114</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">113</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">115</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">114</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 3 and 8 inputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 3 and 8 inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">116</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">115</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">117</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">116</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">118</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">117</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input sequences packed (and potentially padded) into one 3-D</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input sequences packed (and potentially padded) into one 3-D</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">119</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">118</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor with the shape of [seq_length, batch_size, input_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor with the shape of [seq_length, batch_size, input_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">120</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">119</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **W** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **W** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">121</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">120</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The weight tensor for the gates. Concatenation of W[iofc] and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The weight tensor for the gates. Concatenation of W[iofc] and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">122</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">121</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  WB[iofc] (if bidirectional) along dimension 0. The tensor has</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  WB[iofc] (if bidirectional) along dimension 0. The tensor has</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">123</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">122</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  shape [num_directions, 4*hidden_size, input_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  shape [num_directions, 4*hidden_size, input_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">124</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">123</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **R** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **R** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">125</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">124</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The recurrence weight tensor. Concatenation of R[iofc] and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The recurrence weight tensor. Concatenation of R[iofc] and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">126</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">125</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  RB[iofc] (if bidirectional) along dimension 0. This tensor has</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  RB[iofc] (if bidirectional) along dimension 0. This tensor has</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">127</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">126</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  shape [num_directions, 4*hidden_size, hidden_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  shape [num_directions, 4*hidden_size, hidden_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">128</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">127</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">129</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">128</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The bias tensor for input gate. Concatenation of [Wb[iofc],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The bias tensor for input gate. Concatenation of [Wb[iofc],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">130</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">129</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Rb[iofc]], and [WBb[iofc], RBb[iofc]] (if bidirectional) along</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Rb[iofc]], and [WBb[iofc], RBb[iofc]] (if bidirectional) along</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">131</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">130</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimension 0. This tensor has shape [num_directions,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimension 0. This tensor has shape [num_directions,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">132</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">131</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  8*hidden_size]. Optional: If not specified - assumed to be 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  8*hidden_size]. Optional: If not specified - assumed to be 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">133</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">132</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **sequence_lens** (optional, heterogeneous) - **T1**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **sequence_lens** (optional, heterogeneous) - **T1**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">134</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">133</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional tensor specifying lengths of the sequences in a batch. If</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional tensor specifying lengths of the sequences in a batch. If</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">135</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">134</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  not specified - assumed all sequences in the batch to have length</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  not specified - assumed all sequences in the batch to have length</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">136</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">135</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq_length. It has shape [batch_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq_length. It has shape [batch_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">137</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">136</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **initial_h** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **initial_h** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">138</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">137</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional initial value of the hidden. If not specified - assumed to</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional initial value of the hidden. If not specified - assumed to</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">139</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">138</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  be 0. It has shape [num_directions, batch_size, hidden_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  be 0. It has shape [num_directions, batch_size, hidden_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">140</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">139</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **initial_c** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **initial_c** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">141</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">140</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional initial value of the cell. If not specified - assumed to be</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional initial value of the cell. If not specified - assumed to be</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">142</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">141</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  0. It has shape [num_directions, batch_size, hidden_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  0. It has shape [num_directions, batch_size, hidden_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">143</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">142</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **P** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **P** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">144</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">143</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The weight tensor for peepholes. Concatenation of P[iof] and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The weight tensor for peepholes. Concatenation of P[iof] and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">145</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">144</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  PB[iof] (if bidirectional) along dimension 0. It has shape</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  PB[iof] (if bidirectional) along dimension 0. It has shape</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">146</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">145</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  [num_directions, 3*hidde_size]. Optional: If not specified -</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  [num_directions, 3*hidde_size]. Optional: If not specified -</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">147</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">146</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  assumed to be 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  assumed to be 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">148</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">147</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">149</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">148</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">150</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">149</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">151</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">150</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 0 and 3 outputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 0 and 3 outputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">152</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">151</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">153</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">152</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">154</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">153</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A tensor that concats all the intermediate output values of the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A tensor that concats all the intermediate output values of the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">155</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">154</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  hidden. It has shape [seq_length, num_directions, batch_size,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  hidden. It has shape [seq_length, num_directions, batch_size,</code></td></tr>
    <tr style="1px solid black;"><td><code>156</code></td><td><code>155</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  hidden_size].<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">I</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">_</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">q</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  hidden_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">157</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">156</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y_h** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y_h** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">158</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">157</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The last output value of the hidden. It has shape [num_directions,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The last output value of the hidden. It has shape [num_directions,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">159</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">158</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_size, hidden_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_size, hidden_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">160</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">159</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y_c** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y_c** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">161</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">160</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The last output value of the cell. It has shape [num_directions,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The last output value of the cell. It has shape [num_directions,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">162</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">161</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_size, hidden_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_size, hidden_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">163</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">162</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">164</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">163</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">165</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">164</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">166</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">165</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">167</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">166</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">168</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">167</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">169</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">168</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">170</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">169</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">171</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">170</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">172</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">171</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">173</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">172</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">174</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">173</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">175</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">174</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain seq_lens to integer tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain seq_lens to integer tensor.</code></td></tr>
    </table>

.. _l-onnx-op-lstm-1:

LSTM - 1
========

**Version**

* **name**: `LSTM (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#LSTM>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1**.

**Summary**

Computes an one-layer LSTM. This operator is usually supported via some
custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`o` - output gate

`f` - forget gate

`c` - cell gate

`t` - time step (t-1 means previous time step)

`W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates

`R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates

`Wb[iofc]` - W bias vectors for input, output, forget, and cell gates

`Rb[iofc]` - R bias vectors for input, output, forget, and cell gates

`P[iof]`  - P peephole weight vector for input, output, and forget gates

`WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates

`RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates

`WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates

`RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates

`PB[iof]`  - P peephole weight vector for backward input, output, and forget gates

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

  - it = f(Xt*(Wi^T) + Ht-1*Ri + Pi (.) Ct-1 + Wbi + Rbi)

  - ft = f(Xt*(Wf^T) + Ht-1*Rf + Pf (.) Ct-1 + Wbf + Rbf)

  - ct = g(Xt*(Wc^T) + Ht-1*Rc + Wbc + Rbc)

  - Ct = ft (.) Ct-1 + it (.) ct

  - ot = f(Xt*(Wo^T) + Ht-1*Ro + Po (.) Ct + Wbo + Rbo)

  - Ht = ot (.) h(Ct)

**Attributes**

* **activation_alpha**:
  Optional scaling values used by some activation functions. The
  values are consumed in the order of activation functions, for
  example (f, g, h) in LSTM. Default values are the same as of
  corresponding ONNX operators.For example with LeakyRelu, the default
  alpha is 0.01.
* **activation_beta**:
  Optional scaling values used by some activation functions. The
  values are consumed in the order of activation functions, for
  example (f, g, h) in LSTM. Default values are the same as of
  corresponding ONNX operators.
* **activations**:
  A list of 3 (or 6 if bidirectional) activation functions for input,
  output, forget, cell, and hidden. The activation functions must be
  one of the activation functions specified above. Optional: See the
  equations for default if not specified.
* **clip**:
  Cell clip threshold. Clipping bounds the elements of a tensor in the
  range of [-threshold, +threshold] and is applied to the input of
  activations. No clip if not specified.
* **direction**:
  Specify if the RNN is forward, reverse, or bidirectional. Must be
  one of forward (default), reverse, or bidirectional. Default value is ``'forward'``.
* **hidden_size**:
  Number of neurons in the hidden layer
* **input_forget**:
  Couple the input and forget gates if 1, default 0. Default value is ``0``.
* **output_sequence**:
  The sequence output for the hidden is optional if 0. Default 0. Default value is ``0``.

**Inputs**

Between 3 and 8 inputs.

* **X** (heterogeneous) - **T**:
  The input sequences packed (and potentially padded) into one 3-D
  tensor with the shape of `[seq_length, batch_size, input_size]`.
* **W** (heterogeneous) - **T**:
  The weight tensor for the gates. Concatenation of `W[iofc]` and
  `WB[iofc]` (if bidirectional) along dimension 0. The tensor has
  shape `[num_directions, 4*hidden_size, input_size]`.
* **R** (heterogeneous) - **T**:
  The recurrence weight tensor. Concatenation of `R[iofc]` and
  `RB[iofc]` (if bidirectional) along dimension 0. This tensor has
  shape `[num_directions, 4*hidden_size, hidden_size]`.
* **B** (optional, heterogeneous) - **T**:
  The bias tensor for input gate. Concatenation of `[Wb[iofc],
  Rb[iofc]]`, and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along
  dimension 0. This tensor has shape `[num_directions,
  8*hidden_size]`. Optional: If not specified - assumed to be 0.
* **sequence_lens** (optional, heterogeneous) - **T1**:
  Optional tensor specifying lengths of the sequences in a batch. If
  not specified - assumed all sequences in the batch to have length
  `seq_length`. It has shape `[batch_size]`.
* **initial_h** (optional, heterogeneous) - **T**:
  Optional initial value of the hidden. If not specified - assumed to
  be 0. It has shape `[num_directions, batch_size, hidden_size]`.
* **initial_c** (optional, heterogeneous) - **T**:
  Optional initial value of the cell. If not specified - assumed to be
  0. It has shape `[num_directions, batch_size, hidden_size]`.
* **P** (optional, heterogeneous) - **T**:
  The weight tensor for peepholes. Concatenation of `P[iof]` and
  `PB[iof]` (if bidirectional) along dimension 0. It has shape
  `[num_directions, 3*hidde_size]`. Optional: If not specified -
  assumed to be 0.

**Outputs**

Between 0 and 3 outputs.

* **Y** (optional, heterogeneous) - **T**:
  A tensor that concats all the intermediate output values of the
  hidden. It has shape `[seq_length, num_directions, batch_size,
  hidden_size]`. It is optional if `output_sequence` is 0.
* **Y_h** (optional, heterogeneous) - **T**:
  The last output value of the hidden. It has shape `[num_directions,
  batch_size, hidden_size]`.
* **Y_c** (optional, heterogeneous) - **T**:
  The last output value of the cell. It has shape `[num_directions,
  batch_size, hidden_size]`.

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
* **T1** in (
  tensor(int32)
  ):
  Constrain seq_lens to integer tensor.
