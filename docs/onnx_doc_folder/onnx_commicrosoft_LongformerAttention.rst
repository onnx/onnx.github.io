
.. _l-onnx-doccom.microsoft-LongformerAttention:

===================================
com.microsoft - LongformerAttention
===================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-longformerattention-1:

LongformerAttention - 1 (com.microsoft)
=======================================

**Version**

* **name**: `LongformerAttention (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.LongformerAttention>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Longformer Self Attention with a local context and a global context. Tokens attend locally: Each token
attends to its W previous tokens and W succeding tokens with W being the window length. A selected few tokens
attend globally to all other tokens.

The attention mask is of shape (batch_size, sequence_length), where sequence_length is a multiple of 2W after padding.
Mask value < 0 (like -10000.0) means the token is masked, 0 otherwise.

Global attention flags have value 1 for the tokens attend globally and 0 otherwise.

**Attributes**

* **num_heads** (required):
  Number of attention heads Default value is ``?``.
* **window** (required):
  One sided attention windows length W, or half of total window length Default value is ``?``.

**Inputs**

* **input** (heterogeneous) - **T**:
  3D input tensor with shape (batch_size, sequence_length,
  hidden_size), hidden_size = num_heads * head_size
* **weight** (heterogeneous) - **T**:
  2D input tensor with shape (hidden_size, 3 * hidden_size)
* **bias** (heterogeneous) - **T**:
  1D input tensor with shape (3 * hidden_size)
* **mask** (heterogeneous) - **T**:
  Attention mask with shape (batch_size, sequence_length)
* **global_weight** (heterogeneous) - **T**:
  2D input tensor with shape (hidden_size, 3 * hidden_size)
* **global_bias** (heterogeneous) - **T**:
  1D input tensor with shape (3 * hidden_size)
* **global** (heterogeneous) - **G**:
  Global attention flags with shape (batch_size, sequence_length)

**Outputs**

* **output** (heterogeneous) - **T**:
  3D output tensor with shape (batch_size, sequence_length,
  hidden_size)

**Examples**
