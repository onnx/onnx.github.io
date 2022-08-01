
.. _l-onnx-doccom.microsoft-DynamicQuantizeLSTM:

===================================
com.microsoft - DynamicQuantizeLSTM
===================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-dynamicquantizelstm-1:

DynamicQuantizeLSTM - 1 (com.microsoft)
=======================================

**Version**

* **name**: `DynamicQuantizeLSTM (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.DynamicQuantizeLSTM>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

**Attributes**

* **activation_alpha**:
  Optional scaling values used by some activation functions. The
  values are consumed in the order of activation functions, for
  example (f, g, h) in LSTM. Default values are the same as of
  corresponding ONNX operators.For example with LeakyRelu, the default
  alpha is 0.01. Default value is ``?``.
* **activation_beta**:
  Optional scaling values used by some activation functions. The
  values are consumed in the order of activation functions, for
  example (f, g, h) in LSTM. Default values are the same as of
  corresponding ONNX operators. Default value is ``?``.
* **activations**:
  A list of 3 (or 6 if bidirectional) activation functions for input,
  output, forget, cell, and hidden. The activation functions must be
  one of the activation functions specified above. Optional: See the
  equations for default if not specified. Default value is ``?``.
* **clip**:
  Cell clip threshold. Clipping bounds the elements of a tensor in the
  range of [-threshold, +threshold] and is applied to the input of
  activations. No clip if not specified. Default value is ``?``.
* **direction**:
  Specify if the RNN is forward, reverse, or bidirectional. Must be
  one of forward (default), reverse, or bidirectional. Default value is ``?``.
* **hidden_size**:
  Number of neurons in the hidden layer Default value is ``?``.
* **input_forget**:
  Couple the input and forget gates if 1. Default value is ``?``.

**Inputs**

* **X** (heterogeneous) - **T**:
  The input sequences packed (and potentially padded) into one 3-D
  tensor with the shape of `[seq_length, batch_size, input_size]`.
* **W** (heterogeneous) - **T2**:
  The weight tensor for the gates. Concatenation of `W[iofc]` and
  `WB[iofc]` (if bidirectional) along dimension 0. The tensor has
  shape `[num_directions, input_size, 4*hidden_size]`.
* **R** (heterogeneous) - **T2**:
  The recurrence weight tensor. Concatenation of `R[iofc]` and
  `RB[iofc]` (if bidirectional) along dimension 0. This tensor has
  shape `[num_directions, hidden_size, 4*hidden_size]`.
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
* **W_scale** (heterogeneous) - **T**:
  W's scale. Its size is [num_directions] for per-tensor/layer
  quantization, or [num_directions, 4*hidden_size] for per-channel
  quantization on the axis input_size.
* **W_zero_point** (heterogeneous) - **T2**:
  W's zero point. Its size is [num_directions] for per-tensor/layer
  quantization, or [num_directions, 4*hidden_size] for per-channel
  quantization on the axis input_size.
* **R_scale** (heterogeneous) - **T**:
  R's scale. Its size is [num_directions] for per-tensor/layer
  quantization, or [num_directions, 4*hidden_size] for per-channel
  quantization on the axis input_size.
* **R_zero_point** (heterogeneous) - **T2**:
  R's zero point. Its size is [num_directions] for per-tensor/layer
  quantization, or [num_directions, 4*hidden_size] for per-channel
  quantization on the axis input_size.

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

**Examples**
