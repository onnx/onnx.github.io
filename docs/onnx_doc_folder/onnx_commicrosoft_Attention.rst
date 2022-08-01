
.. _l-onnx-doccom.microsoft-Attention:

=========================
com.microsoft - Attention
=========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-attention-1:

Attention - 1 (com.microsoft)
=============================

**Version**

* **name**: `Attention (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.Attention>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Multi-Head Self Attention that can be either unidirectional (like GPT-2) or bidirectional (like BERT).
The mask_index input is optional. Besides raw attention mask with shape (batch_size, past_sequence_length + sequence_length)
or (batch_size, sequence_length, past_sequence_length + sequence_length) with value 0 for masked and 1 otherwise,
we also support other two formats: When input has right-side padding, mask_index is one dimension with shape (batch_size),
where value of each element is the end position, or valid length of actual sequence excluding padding. When input has
left-side padding, mask_index has shape (2 * batch_size), where the values are the exclusive end positions followed by
the inclusive start positions. When unidirectional is 1, and each token only attend to previous tokens. For GPT-2, both past
and present state are optional. Present state could appear in output even when past state is not in input.

**Attributes**

* **num_heads** (required):
  Number of attention heads Default value is ``?``.
* **qkv_hidden_sizes**:
  Hidden layer sizes of Q, K, V paths in Attention Default value is ``?``.
* **unidirectional**:
  Whether every token can only attend to previous tokens. Default
  value is 0. Default value is ``?``.

**Inputs**

Between 3 and 6 inputs.

* **input** (heterogeneous) - **T**:
  3D input tensor with shape (batch_size, sequence_length,
  input_hidden_size)
* **weight** (heterogeneous) - **T**:
  2D input tensor with shape (input_hidden_size, 3 * hidden_size),
  where hidden_size = num_heads * head_size
* **bias** (heterogeneous) - **T**:
  1D input tensor with shape (3 * hidden_size)
* **mask_index** (optional, heterogeneous) - **M**:
  Attention mask with shape (batch_size, 1, max_sequence_length,
  max_sequence_length), (batch_size, past_sequence_length +
  sequence_length)or (batch_size, sequence_length,
  past_sequence_length + sequence_length), or index with shape
  (batch_size) or (2 * batch_size).
* **past** (optional, heterogeneous) - **T**:
  past state for key and value with shape (2, batch_size, num_heads,
  past_sequence_length, head_size).
* **extra_add** (optional, heterogeneous) - **T**:
  additional add to QxK' with shape (batch_size, num_heads,
  sequence_length, sequence_length).

**Outputs**

Between 1 and 2 outputs.

* **output** (heterogeneous) - **T**:
  3D output tensor with shape (batch_size, sequence_length,
  hidden_size)
* **present** (optional, heterogeneous) - **T**:
  present state for key and value with shape (2, batch_size,
  num_heads, past_sequence_length + sequence_length, head_size)

**Examples**
