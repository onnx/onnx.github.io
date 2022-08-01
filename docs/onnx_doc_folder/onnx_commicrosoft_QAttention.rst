
.. _l-onnx-doccom.microsoft-QAttention:

==========================
com.microsoft - QAttention
==========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-qattention-1:

QAttention - 1 (com.microsoft)
==============================

**Version**

* **name**: `QAttention (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.QAttention>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Quantization of Multi-Head Self Attention.

**Attributes**

* **num_heads** (required):
  Number of attention heads Default value is ``?``.
* **unidirectional**:
  Whether every token can only attend to previous tokens. Default
  value is 0. Default value is ``?``.

**Inputs**

Between 5 and 9 inputs.

* **input** (heterogeneous) - **T1**:
  3D input tensor with shape (batch_size, sequence_length,
  input_hidden_size)
* **weight** (heterogeneous) - **T2**:
  2D input tensor with shape (input_hidden_size, 3 * hidden_size),
  hidden_size = num_heads * head_size
* **bias** (heterogeneous) - **T3**:
  1D input tensor with shape (3 * hidden_size)
* **input_scale** (heterogeneous) - **T3**:
  scale of quantized input tensor. It's a scalar, which means a per-
  tensor/layer quantization.
* **weight_scale** (heterogeneous) - **T3**:
  scale of weight scale. It's a scalar or a 1D tensor, which means a
  per-tensor/per-column quantization.Its size should be 3 *
  hidden_size if it is per-column quantization
* **mask_index** (optional, heterogeneous) - **T4**:
  Attention mask index with shape (batch_size)
* **input_zero_point** (optional, heterogeneous) - **T1**:
  zero point of quantized input tensor.It's a scalar, which means a
  per-tensor/layer quantization.
* **weight_zero_point** (optional, heterogeneous) - **T2**:
  zero point of quantized weight tensor. It's a scalar or a 1D tensor,
  which means a per-tensor/per-column quantization.Its size should be
  3 * hidden_size if it is per-column quantization
* **past** (optional, heterogeneous) - **T3**:
  past state for key and value with shape (2, batch_size, num_heads,
  past_sequence_length, head_size).

**Outputs**

Between 1 and 2 outputs.

* **output** (heterogeneous) - **T3**:
  3D output tensor with shape (batch_size, sequence_length,
  hidden_size)
* **present** (optional, heterogeneous) - **T3**:
  present state for key and value with shape (2, batch_size,
  num_heads, past_sequence_length + sequence_length, head_size)

**Examples**
