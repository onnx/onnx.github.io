
.. _l-onnx-doc-RNN:

===
RNN
===

.. contents::
    :local:


.. _l-onnx-op-rnn-14:

RNN - 14
========

**Version**

* **name**: `RNN (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#RNN>`_
* **domain**: **main**
* **since_version**: **14**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 14**.

**Summary**

Computes an one-layer simple RNN. This operator is usually supported
via some custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`t` - time step (t-1 means previous time step)

`Wi` - W parameter weight matrix for input gate

`Ri` - R recurrence weight matrix for input gate

`Wbi` - W parameter bias vector for input gate

`Rbi` - R parameter bias vector for input gate

`WBi` - W parameter weight matrix for backward input gate

`RBi` - R recurrence weight matrix for backward input gate

`WBbi` - WR bias vectors for backward input gate

`RBbi` - RR bias vectors for backward input gate

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

Equations (Default: f=Tanh):

  - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
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
  One (or two if bidirectional) activation function for input gate.
  The activation function must be one of the activation functions
  specified above. Optional: Default `Tanh` if not specified. Default value is ``[b'Tanh' b'Tanh']``.
* **clip**:
  Cell clip threshold. Clipping bounds the elements of a tensor in the
  range of [-threshold, +threshold] and is applied to the input of
  activations. No clip if not specified.
* **direction**:
  Specify if the RNN is forward, reverse, or bidirectional. Must be
  one of forward (default), reverse, or bidirectional. Default value is ``'forward'``.
* **hidden_size**:
  Number of neurons in the hidden layer
* **layout**:
  The shape format of inputs X, initial_h and outputs Y, Y_h. If 0,
  the following shapes are expected: X.shape = [seq_length,
  batch_size, input_size], Y.shape = [seq_length, num_directions,
  batch_size, hidden_size], initial_h.shape = Y_h.shape =
  [num_directions, batch_size, hidden_size]. If 1, the following
  shapes are expected: X.shape = [batch_size, seq_length, input_size],
  Y.shape = [batch_size, seq_length, num_directions, hidden_size],
  initial_h.shape = Y_h.shape = [batch_size, num_directions,
  hidden_size]. Default value is ``0``.

**Inputs**

Between 3 and 6 inputs.

* **X** (heterogeneous) - **T**:
  The input sequences packed (and potentially padded) into one 3-D
  tensor with the shape of `[seq_length, batch_size, input_size]`.
* **W** (heterogeneous) - **T**:
  The weight tensor for input gate. Concatenation of `Wi` and `WBi`
  (if bidirectional). The tensor has shape `[num_directions,
  hidden_size, input_size]`.
* **R** (heterogeneous) - **T**:
  The recurrence weight tensor. Concatenation of `Ri` and `RBi` (if
  bidirectional). The tensor has shape `[num_directions, hidden_size,
  hidden_size]`.
* **B** (optional, heterogeneous) - **T**:
  The bias tensor for input gate. Concatenation of `[Wbi, Rbi]` and
  `[WBbi, RBbi]` (if bidirectional). The tensor has shape
  `[num_directions, 2*hidden_size]`. Optional: If not specified -
  assumed to be 0.
* **sequence_lens** (optional, heterogeneous) - **T1**:
  Optional tensor specifying lengths of the sequences in a batch. If
  not specified - assumed all sequences in the batch to have length
  `seq_length`. It has shape `[batch_size]`.
* **initial_h** (optional, heterogeneous) - **T**:
  Optional initial value of the hidden. If not specified - assumed to
  be 0. It has shape `[num_directions, batch_size, hidden_size]`.

**Outputs**

Between 0 and 2 outputs.

* **Y** (optional, heterogeneous) - **T**:
  A tensor that concats all the intermediate output values of the
  hidden. It has shape `[seq_length, num_directions, batch_size,
  hidden_size]`.
* **Y_h** (optional, heterogeneous) - **T**:
  The last output value of the hidden. It has shape `[num_directions,
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
    hidden_size = 4
    weight_scale = 0.1

    node = onnx.helper.make_node(
        'RNN',
        inputs=['X', 'W', 'R'],
        outputs=['', 'Y_h'],
        hidden_size=hidden_size
    )

    W = weight_scale * np.ones((1, hidden_size, input_size)).astype(np.float32)
    R = weight_scale * np.ones((1, hidden_size, hidden_size)).astype(np.float32)

    rnn = RNN_Helper(X=input, W=W, R=R)
    _, Y_h = rnn.step()
    expect(node, inputs=[input, W, R], outputs=[Y_h.astype(np.float32)], name='test_simple_rnn_defaults')

**initial_bias**

::

    input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]).astype(np.float32)

    input_size = 3
    hidden_size = 5
    custom_bias = 0.1
    weight_scale = 0.1

    node = onnx.helper.make_node(
        'RNN',
        inputs=['X', 'W', 'R', 'B'],
        outputs=['', 'Y_h'],
        hidden_size=hidden_size
    )

    W = weight_scale * np.ones((1, hidden_size, input_size)).astype(np.float32)
    R = weight_scale * np.ones((1, hidden_size, hidden_size)).astype(np.float32)

    # Adding custom bias
    W_B = custom_bias * np.ones((1, hidden_size)).astype(np.float32)
    R_B = np.zeros((1, hidden_size)).astype(np.float32)
    B = np.concatenate((W_B, R_B), axis=1)

    rnn = RNN_Helper(X=input, W=W, R=R, B=B)
    _, Y_h = rnn.step()
    expect(node, inputs=[input, W, R, B], outputs=[Y_h.astype(np.float32)],
           name='test_simple_rnn_with_initial_bias')

**seq_length**

::

    input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                      [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]]]).astype(np.float32)

    input_size = 3
    hidden_size = 5

    node = onnx.helper.make_node(
        'RNN',
        inputs=['X', 'W', 'R', 'B'],
        outputs=['', 'Y_h'],
        hidden_size=hidden_size
    )

    W = np.random.randn(1, hidden_size, input_size).astype(np.float32)
    R = np.random.randn(1, hidden_size, hidden_size).astype(np.float32)

    # Adding custom bias
    W_B = np.random.randn(1, hidden_size).astype(np.float32)
    R_B = np.random.randn(1, hidden_size).astype(np.float32)
    B = np.concatenate((W_B, R_B), axis=1)

    rnn = RNN_Helper(X=input, W=W, R=R, B=B)
    _, Y_h = rnn.step()
    expect(node, inputs=[input, W, R, B], outputs=[Y_h.astype(np.float32)], name='test_rnn_seq_length')

**batchwise**

::

    input = np.array([[[1., 2.]], [[3., 4.]], [[5., 6.]]]).astype(np.float32)

    input_size = 2
    hidden_size = 4
    weight_scale = 0.5
    layout = 1

    node = onnx.helper.make_node(
        'RNN',
        inputs=['X', 'W', 'R'],
        outputs=['Y', 'Y_h'],
        hidden_size=hidden_size,
        layout=layout
    )

    W = weight_scale * np.ones((1, hidden_size, input_size)).astype(np.float32)
    R = weight_scale * np.ones((1, hidden_size, hidden_size)).astype(np.float32)

    rnn = RNN_Helper(X=input, W=W, R=R, layout=layout)
    Y, Y_h = rnn.step()
    expect(node, inputs=[input, W, R], outputs=[Y.astype(np.float32), Y_h.astype(np.float32)], name='test_simple_rnn_batchwise')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Computes an one-layer simple RNN. This operator is usually supported</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Computes an one-layer simple RNN. This operator is usually supported</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">via some custom implementation such as CuDNN.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">via some custom implementation such as CuDNN.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Notations:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Notations:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">X - input tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">X - input tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">i - input gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">i - input gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">t - time step (t-1 means previous time step)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">t - time step (t-1 means previous time step)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Wi - W parameter weight matrix for input gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Wi - W parameter weight matrix for input gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Ri - R recurrence weight matrix for input gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Ri - R recurrence weight matrix for input gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Wbi - W parameter bias vector for input gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Wbi - W parameter bias vector for input gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Rbi - R parameter bias vector for input gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Rbi - R parameter bias vector for input gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">WBi - W parameter weight matrix for backward input gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">WBi - W parameter weight matrix for backward input gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">RBi - R recurrence weight matrix for backward input gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">RBi - R recurrence weight matrix for backward input gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">WBbi - WR bias vectors for backward input gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">WBbi - WR bias vectors for backward input gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">RBbi - RR bias vectors for backward input gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">RBbi - RR bias vectors for backward input gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">H - Hidden state</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">H - Hidden state</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">num_directions - 2 if direction == bidirectional else 1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">num_directions - 2 if direction == bidirectional else 1</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Activation functions:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Activation functions:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Relu(x)                - max(0, x)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Relu(x)                - max(0, x)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Sigmoid(x)             - 1/(1 + e^{-x})</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Sigmoid(x)             - 1/(1 + e^{-x})</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (NOTE: Below are optional)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (NOTE: Below are optional)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Affine(x)              - alpha*x + beta</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Affine(x)              - alpha*x + beta</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  LeakyRelu(x)           - x if x >= 0 else alpha * x</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  LeakyRelu(x)           - x if x >= 0 else alpha * x</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ThresholdedRelu(x)     - x if x >= alpha else 0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ThresholdedRelu(x)     - x if x >= alpha else 0</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ScaledTanh(x)          - alpha*Tanh(beta*x)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ScaledTanh(x)          - alpha*Tanh(beta*x)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Softsign(x)            - x/(1 + |x|)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Softsign(x)            - x/(1 + |x|)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Softplus(x)            - log(1 + e^x)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Softplus(x)            - log(1 + e^x)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Equations (Default: f=Tanh):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Equations (Default: f=Tanh):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator has **optional** inputs/outputs. See ONNX <https://github.com/onnx/onnx/blob/master/docs/IR.md>_ for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator has **optional** inputs/outputs. See ONNX <https://github.com/onnx/onnx/blob/master/docs/IR.md>_ for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activation_alpha**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activation_alpha**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional scaling values used by some activation functions. The</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional scaling values used by some activation functions. The</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values are consumed in the order of activation functions, for</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values are consumed in the order of activation functions, for</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  example (f, g, h) in LSTM. Default values are the same as of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  example (f, g, h) in LSTM. Default values are the same as of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding ONNX operators.For example with LeakyRelu, the default</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding ONNX operators.For example with LeakyRelu, the default</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  alpha is 0.01.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  alpha is 0.01.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activation_beta**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activation_beta**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional scaling values used by some activation functions. The</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional scaling values used by some activation functions. The</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values are consumed in the order of activation functions, for</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values are consumed in the order of activation functions, for</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  example (f, g, h) in LSTM. Default values are the same as of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  example (f, g, h) in LSTM. Default values are the same as of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding ONNX operators.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding ONNX operators.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activations**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activations**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  One (or two if bidirectional) activation function for input gate.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  One (or two if bidirectional) activation function for input gate.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The activation function must be one of the activation functions</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The activation function must be one of the activation functions</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  specified above. Optional: Default Tanh if not specified. Default value is [b'Tanh' b'Tanh'].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  specified above. Optional: Default Tanh if not specified. Default value is [b'Tanh' b'Tanh'].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **clip**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **clip**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Cell clip threshold. Clipping bounds the elements of a tensor in the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Cell clip threshold. Clipping bounds the elements of a tensor in the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  range of [-threshold, +threshold] and is applied to the input of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  range of [-threshold, +threshold] and is applied to the input of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  activations. No clip if not specified.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  activations. No clip if not specified.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **direction**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **direction**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Specify if the RNN is forward, reverse, or bidirectional. Must be</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Specify if the RNN is forward, reverse, or bidirectional. Must be</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  one of forward (default), reverse, or bidirectional. Default value is 'forward'.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  one of forward (default), reverse, or bidirectional. Default value is 'forward'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **hidden_size**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **hidden_size**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Number of neurons in the hidden layer</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Number of neurons in the hidden layer</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">88</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **layout**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">89</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  The shape format of inputs X, initial_h and outputs Y, Y_h. If 0,</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">90</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  the following shapes are expected: X.shape = [seq_length,</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">91</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  batch_size, input_size], Y.shape = [seq_length, num_directions,</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">92</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  batch_size, hidden_size], initial_h.shape = Y_h.shape =</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">93</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  [num_directions, batch_size, hidden_size]. If 1, the following</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">94</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  shapes are expected: X.shape = [batch_size, seq_length, input_size],</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">95</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Y.shape = [batch_size, seq_length, num_directions, hidden_size],</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">96</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  initial_h.shape = Y_h.shape = [batch_size, num_directions,</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">97</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  hidden_size]. Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 3 and 6 inputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 3 and 6 inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input sequences packed (and potentially padded) into one 3-D</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input sequences packed (and potentially padded) into one 3-D</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor with the shape of [seq_length, batch_size, input_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor with the shape of [seq_length, batch_size, input_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **W** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **W** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The weight tensor for input gate. Concatenation of Wi and WBi</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The weight tensor for input gate. Concatenation of Wi and WBi</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (if bidirectional). The tensor has shape [num_directions,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (if bidirectional). The tensor has shape [num_directions,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">109</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  hidden_size, input_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  hidden_size, input_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">110</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **R** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **R** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">111</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The recurrence weight tensor. Concatenation of Ri and RBi (if</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The recurrence weight tensor. Concatenation of Ri and RBi (if</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">112</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  bidirectional). The tensor has shape [num_directions, hidden_size,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  bidirectional). The tensor has shape [num_directions, hidden_size,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">113</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  hidden_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  hidden_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">114</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">115</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The bias tensor for input gate. Concatenation of [Wbi, Rbi] and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The bias tensor for input gate. Concatenation of [Wbi, Rbi] and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">116</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  [WBbi, RBbi] (if bidirectional). The tensor has shape</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  [WBbi, RBbi] (if bidirectional). The tensor has shape</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">117</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  [num_directions, 2*hidden_size]. Optional: If not specified -</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  [num_directions, 2*hidden_size]. Optional: If not specified -</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">118</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  assumed to be 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  assumed to be 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">109</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">119</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **sequence_lens** (optional, heterogeneous) - **T1**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **sequence_lens** (optional, heterogeneous) - **T1**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">110</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">120</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional tensor specifying lengths of the sequences in a batch. If</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional tensor specifying lengths of the sequences in a batch. If</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">111</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">121</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  not specified - assumed all sequences in the batch to have length</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  not specified - assumed all sequences in the batch to have length</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">112</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">122</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq_length. It has shape [batch_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq_length. It has shape [batch_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">113</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">123</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **initial_h** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **initial_h** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">114</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">124</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional initial value of the hidden. If not specified - assumed to</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional initial value of the hidden. If not specified - assumed to</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">115</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">125</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  be 0. It has shape [num_directions, batch_size, hidden_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  be 0. It has shape [num_directions, batch_size, hidden_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">116</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">126</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">117</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">127</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">118</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">128</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">119</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">129</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 0 and 2 outputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 0 and 2 outputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">120</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">130</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">121</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">131</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">122</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">132</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A tensor that concats all the intermediate output values of the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A tensor that concats all the intermediate output values of the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">123</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">133</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  hidden. It has shape [seq_length, num_directions, batch_size,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  hidden. It has shape [seq_length, num_directions, batch_size,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">124</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">134</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  hidden_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  hidden_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">125</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">135</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y_h** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y_h** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">126</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">136</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The last output value of the hidden. It has shape [num_directions,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The last output value of the hidden. It has shape [num_directions,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">127</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">137</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_size, hidden_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_size, hidden_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">128</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">138</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">129</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">139</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">130</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">140</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">131</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">141</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">132</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">142</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">133</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">143</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">134</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">144</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">135</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">145</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">136</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">146</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">137</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">147</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">138</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">148</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">139</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">149</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">140</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">150</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain seq_lens to integer tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain seq_lens to integer tensor.</code></td></tr>
    </table>

.. _l-onnx-op-rnn-7:

RNN - 7
=======

**Version**

* **name**: `RNN (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#RNN>`_
* **domain**: **main**
* **since_version**: **7**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 7**.

**Summary**

Computes an one-layer simple RNN. This operator is usually supported
via some custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`t` - time step (t-1 means previous time step)

`Wi` - W parameter weight matrix for input gate

`Ri` - R recurrence weight matrix for input gate

`Wbi` - W parameter bias vector for input gate

`Rbi` - R parameter bias vector for input gate

`WBi` - W parameter weight matrix for backward input gate

`RBi` - R recurrence weight matrix for backward input gate

`WBbi` - WR bias vectors for backward input gate

`RBbi` - RR bias vectors for backward input gate

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

Equations (Default: f=Tanh):

  - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
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
  One (or two if bidirectional) activation function for input gate.
  The activation function must be one of the activation functions
  specified above. Optional: Default `Tanh` if not specified. Default value is ``[b'Tanh' b'Tanh']``.
* **clip**:
  Cell clip threshold. Clipping bounds the elements of a tensor in the
  range of [-threshold, +threshold] and is applied to the input of
  activations. No clip if not specified.
* **direction**:
  Specify if the RNN is forward, reverse, or bidirectional. Must be
  one of forward (default), reverse, or bidirectional. Default value is ``'forward'``.
* **hidden_size**:
  Number of neurons in the hidden layer

**Inputs**

Between 3 and 6 inputs.

* **X** (heterogeneous) - **T**:
  The input sequences packed (and potentially padded) into one 3-D
  tensor with the shape of `[seq_length, batch_size, input_size]`.
* **W** (heterogeneous) - **T**:
  The weight tensor for input gate. Concatenation of `Wi` and `WBi`
  (if bidirectional). The tensor has shape `[num_directions,
  hidden_size, input_size]`.
* **R** (heterogeneous) - **T**:
  The recurrence weight tensor. Concatenation of `Ri` and `RBi` (if
  bidirectional). The tensor has shape `[num_directions, hidden_size,
  hidden_size]`.
* **B** (optional, heterogeneous) - **T**:
  The bias tensor for input gate. Concatenation of `[Wbi, Rbi]` and
  `[WBbi, RBbi]` (if bidirectional). The tensor has shape
  `[num_directions, 2*hidden_size]`. Optional: If not specified -
  assumed to be 0.
* **sequence_lens** (optional, heterogeneous) - **T1**:
  Optional tensor specifying lengths of the sequences in a batch. If
  not specified - assumed all sequences in the batch to have length
  `seq_length`. It has shape `[batch_size]`.
* **initial_h** (optional, heterogeneous) - **T**:
  Optional initial value of the hidden. If not specified - assumed to
  be 0. It has shape `[num_directions, batch_size, hidden_size]`.

**Outputs**

Between 0 and 2 outputs.

* **Y** (optional, heterogeneous) - **T**:
  A tensor that concats all the intermediate output values of the
  hidden. It has shape `[seq_length, num_directions, batch_size,
  hidden_size]`.
* **Y_h** (optional, heterogeneous) - **T**:
  The last output value of the hidden. It has shape `[num_directions,
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
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Computes an one-layer simple RNN. This operator is usually supported</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Computes an one-layer simple RNN. This operator is usually supported</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">via some custom implementation such as CuDNN.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">via some custom implementation such as CuDNN.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Notations:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Notations:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">X - input tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">X - input tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">i - input gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">i - input gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">t - time step (t-1 means previous time step)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">t - time step (t-1 means previous time step)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Wi - W parameter weight matrix for input gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Wi - W parameter weight matrix for input gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Ri - R recurrence weight matrix for input gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Ri - R recurrence weight matrix for input gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Wbi - W parameter bias vector for input gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Wbi - W parameter bias vector for input gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Rbi - R parameter bias vector for input gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Rbi - R parameter bias vector for input gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">WBi - W parameter weight matrix for backward input gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">WBi - W parameter weight matrix for backward input gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">RBi - R recurrence weight matrix for backward input gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">RBi - R recurrence weight matrix for backward input gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">WBbi - WR bias vectors for backward input gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">WBbi - WR bias vectors for backward input gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">RBbi - RR bias vectors for backward input gate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">RBbi - RR bias vectors for backward input gate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">H - Hidden state</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">H - Hidden state</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">num_directions - 2 if direction == bidirectional else 1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">num_directions - 2 if direction == bidirectional else 1</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Activation functions:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Activation functions:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Relu(x)                - max(0, x)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Relu(x)                - max(0, x)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Sigmoid(x)             - 1/(1 + e^{-x})</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Sigmoid(x)             - 1/(1 + e^{-x})</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (NOTE: Below are optional)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (NOTE: Below are optional)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Affine(x)              - alpha*x + beta</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Affine(x)              - alpha*x + beta</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  LeakyRelu(x)           - x if x >= 0 else alpha * x</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  LeakyRelu(x)           - x if x >= 0 else alpha * x</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ThresholdedRelu(x)     - x if x >= alpha else 0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ThresholdedRelu(x)     - x if x >= alpha else 0</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ScaledTanh(x)          - alpha*Tanh(beta*x)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ScaledTanh(x)          - alpha*Tanh(beta*x)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Softsign(x)            - x/(1 + |x|)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Softsign(x)            - x/(1 + |x|)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Softplus(x)            - log(1 + e^x)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Softplus(x)            - log(1 + e^x)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Equations (Default: f=Tanh):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Equations (Default: f=Tanh):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">59</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)</code></td></tr>
    <tr style="1px solid black;"><td><code>59</code></td><td><code>60</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  <span style="color:#BA4A00;">-</span> <span style="color:#BA4A00;">H</span>t <span style="color:#BA4A00;">=</span> <span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">(</span>Xt<span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">(</span><span style="color:#BA4A00;">W</span>i<span style="color:#BA4A00;">^</span><span style="color:#BA4A00;">T</span><span style="color:#BA4A00;">)</span> <span style="color:#BA4A00;">+</span> <span style="color:#BA4A00;">H</span>t<span style="color:#BA4A00;">-</span><span style="color:#BA4A00;">1</span><span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">R</span>i <span style="color:#BA4A00;">+</span> <span style="color:#BA4A00;">W</span>bi <span style="color:#BA4A00;">+</span> <span style="color:#BA4A00;">R</span>bi)</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">T</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span> <span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span> <span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">s</span> <span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span>t<span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span> <span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">.</span> <span style="color:#196F3D;">S</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">O</span><span style="color:#196F3D;">N</span><span style="color:#196F3D;">N</span>X<span style="color:#196F3D;"> </span><span style="color:#196F3D;"><</span><span style="color:#196F3D;">h</span>t<span style="color:#196F3D;">t</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">:</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">g</span>i<span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">I</span><span style="color:#196F3D;">R</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">></span><span style="color:#196F3D;">_</span> <span style="color:#196F3D;">f</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span> <span style="color:#196F3D;">m</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span>t<span style="color:#196F3D;">a</span>i<span style="color:#196F3D;">l</span><span style="color:#196F3D;">s</span> <span style="color:#196F3D;">a</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span> <span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">A</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span>b<span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span>i<span style="color:#196F3D;">n</span> <span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span> <span style="color:#196F3D;">p</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">'</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">T</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">w</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span>b<span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span>i<span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span>)<span style="color:#196F3D;"> </span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activation_alpha**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activation_alpha**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional scaling values used by some activation functions. The</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional scaling values used by some activation functions. The</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values are consumed in the order of activation functions, for</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values are consumed in the order of activation functions, for</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  example (f, g, h) in LSTM. Default values are the same as of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  example (f, g, h) in LSTM. Default values are the same as of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding ONNX operators.For example with LeakyRelu, the default</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding ONNX operators.For example with LeakyRelu, the default</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  alpha is 0.01.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  alpha is 0.01.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activation_beta**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activation_beta**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional scaling values used by some activation functions. The</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional scaling values used by some activation functions. The</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values are consumed in the order of activation functions, for</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values are consumed in the order of activation functions, for</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  example (f, g, h) in LSTM. Default values are the same as of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  example (f, g, h) in LSTM. Default values are the same as of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding ONNX operators.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding ONNX operators.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activations**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **activations**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  One (or two if bidirectional) activation function for input gate.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  One (or two if bidirectional) activation function for input gate.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The activation function must be one of the activation functions</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The activation function must be one of the activation functions</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  specified above. Optional: Default Tanh if not specified. Default value is [b'Tanh' b'Tanh'].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  specified above. Optional: Default Tanh if not specified. Default value is [b'Tanh' b'Tanh'].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **clip**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **clip**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Cell clip threshold. Clipping bounds the elements of a tensor in the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Cell clip threshold. Clipping bounds the elements of a tensor in the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  range of [-threshold, +threshold] and is applied to the input of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  range of [-threshold, +threshold] and is applied to the input of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  activations. No clip if not specified.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  activations. No clip if not specified.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **direction**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **direction**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Specify if the RNN is forward, reverse, or bidirectional. Must be</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Specify if the RNN is forward, reverse, or bidirectional. Must be</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  one of forward (default), reverse, or bidirectional. Default value is 'forward'.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  one of forward (default), reverse, or bidirectional. Default value is 'forward'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **hidden_size**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **hidden_size**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Number of neurons in the hidden layer</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Number of neurons in the hidden layer</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">87</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">* **output_sequence**:</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">88</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  The sequence output for the hidden is optional if 0. Default 0. Default value is 0.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 3 and 6 inputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 3 and 6 inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input sequences packed (and potentially padded) into one 3-D</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input sequences packed (and potentially padded) into one 3-D</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor with the shape of [seq_length, batch_size, input_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor with the shape of [seq_length, batch_size, input_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **W** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **W** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The weight tensor for input gate. Concatenation of Wi and WBi</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The weight tensor for input gate. Concatenation of Wi and WBi</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (if bidirectional). The tensor has shape [num_directions,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (if bidirectional). The tensor has shape [num_directions,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  hidden_size, input_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  hidden_size, input_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **R** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **R** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The recurrence weight tensor. Concatenation of Ri and RBi (if</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The recurrence weight tensor. Concatenation of Ri and RBi (if</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  bidirectional). The tensor has shape [num_directions, hidden_size,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  bidirectional). The tensor has shape [num_directions, hidden_size,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  hidden_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  hidden_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The bias tensor for input gate. Concatenation of [Wbi, Rbi] and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The bias tensor for input gate. Concatenation of [Wbi, Rbi] and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  [WBbi, RBbi] (if bidirectional). The tensor has shape</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  [WBbi, RBbi] (if bidirectional). The tensor has shape</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  [num_directions, 2*hidden_size]. Optional: If not specified -</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  [num_directions, 2*hidden_size]. Optional: If not specified -</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">109</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  assumed to be 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  assumed to be 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">110</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">109</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **sequence_lens** (optional, heterogeneous) - **T1**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **sequence_lens** (optional, heterogeneous) - **T1**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">111</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">110</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional tensor specifying lengths of the sequences in a batch. If</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional tensor specifying lengths of the sequences in a batch. If</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">112</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">111</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  not specified - assumed all sequences in the batch to have length</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  not specified - assumed all sequences in the batch to have length</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">113</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">112</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq_length. It has shape [batch_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq_length. It has shape [batch_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">114</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">113</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **initial_h** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **initial_h** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">115</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">114</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional initial value of the hidden. If not specified - assumed to</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional initial value of the hidden. If not specified - assumed to</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">116</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">115</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  be 0. It has shape [num_directions, batch_size, hidden_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  be 0. It has shape [num_directions, batch_size, hidden_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">117</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">116</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">118</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">117</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">119</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">118</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">120</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">119</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 0 and 2 outputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 0 and 2 outputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">121</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">120</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">122</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">121</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">123</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">122</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A tensor that concats all the intermediate output values of the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A tensor that concats all the intermediate output values of the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">124</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">123</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  hidden. It has shape [seq_length, num_directions, batch_size,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  hidden. It has shape [seq_length, num_directions, batch_size,</code></td></tr>
    <tr style="1px solid black;"><td><code>125</code></td><td><code>124</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  hidden_size].<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">I</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">_</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">q</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  hidden_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">126</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">125</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y_h** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y_h** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">127</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">126</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The last output value of the hidden. It has shape [num_directions,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The last output value of the hidden. It has shape [num_directions,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">128</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">127</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_size, hidden_size].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch_size, hidden_size].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">129</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">128</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">130</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">129</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">131</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">130</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">132</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">131</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">133</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">132</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">134</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">133</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">135</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">134</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">136</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">135</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">137</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">136</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">138</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">137</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">139</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">138</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">140</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">139</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">141</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">140</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain seq_lens to integer tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain seq_lens to integer tensor.</code></td></tr>
    </table>

.. _l-onnx-op-rnn-1:

RNN - 1
=======

**Version**

* **name**: `RNN (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#RNN>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1**.

**Summary**

Computes an one-layer simple RNN. This operator is usually supported
via some custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`t` - time step (t-1 means previous time step)

`Wi` - W parameter weight matrix for input gate

`Ri` - R recurrence weight matrix for input gate

`Wbi` - W parameter bias vector for input gate

`Rbi` - R parameter bias vector for input gate

`WBi` - W parameter weight matrix for backward input gate

`RBi` - R recurrence weight matrix for backward input gate

`WBbi` - WR bias vectors for backward input gate

`RBbi` - RR bias vectors for backward input gate

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

Equations (Default: f=Tanh):

  - Ht = f(Xt*(Wi^T) + Ht-1*Ri + Wbi + Rbi)

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
  One (or two if bidirectional) activation function for input gate.
  The activation function must be one of the activation functions
  specified above. Optional: Default `Tanh` if not specified. Default value is ``[b'Tanh' b'Tanh']``.
* **clip**:
  Cell clip threshold. Clipping bounds the elements of a tensor in the
  range of [-threshold, +threshold] and is applied to the input of
  activations. No clip if not specified.
* **direction**:
  Specify if the RNN is forward, reverse, or bidirectional. Must be
  one of forward (default), reverse, or bidirectional. Default value is ``'forward'``.
* **hidden_size**:
  Number of neurons in the hidden layer
* **output_sequence**:
  The sequence output for the hidden is optional if 0. Default 0. Default value is ``0``.

**Inputs**

Between 3 and 6 inputs.

* **X** (heterogeneous) - **T**:
  The input sequences packed (and potentially padded) into one 3-D
  tensor with the shape of `[seq_length, batch_size, input_size]`.
* **W** (heterogeneous) - **T**:
  The weight tensor for input gate. Concatenation of `Wi` and `WBi`
  (if bidirectional). The tensor has shape `[num_directions,
  hidden_size, input_size]`.
* **R** (heterogeneous) - **T**:
  The recurrence weight tensor. Concatenation of `Ri` and `RBi` (if
  bidirectional). The tensor has shape `[num_directions, hidden_size,
  hidden_size]`.
* **B** (optional, heterogeneous) - **T**:
  The bias tensor for input gate. Concatenation of `[Wbi, Rbi]` and
  `[WBbi, RBbi]` (if bidirectional). The tensor has shape
  `[num_directions, 2*hidden_size]`. Optional: If not specified -
  assumed to be 0.
* **sequence_lens** (optional, heterogeneous) - **T1**:
  Optional tensor specifying lengths of the sequences in a batch. If
  not specified - assumed all sequences in the batch to have length
  `seq_length`. It has shape `[batch_size]`.
* **initial_h** (optional, heterogeneous) - **T**:
  Optional initial value of the hidden. If not specified - assumed to
  be 0. It has shape `[num_directions, batch_size, hidden_size]`.

**Outputs**

Between 0 and 2 outputs.

* **Y** (optional, heterogeneous) - **T**:
  A tensor that concats all the intermediate output values of the
  hidden. It has shape `[seq_length, num_directions, batch_size,
  hidden_size]`. It is optional if `output_sequence` is 0.
* **Y_h** (optional, heterogeneous) - **T**:
  The last output value of the hidden. It has shape `[num_directions,
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
