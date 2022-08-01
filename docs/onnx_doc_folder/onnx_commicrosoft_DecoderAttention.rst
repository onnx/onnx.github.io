
.. _l-onnx-doccom.microsoft-DecoderAttention:

================================
com.microsoft - DecoderAttention
================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-decoderattention-1:

DecoderAttention - 1 (com.microsoft)
====================================

**Version**

* **name**: `DecoderAttention (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.DecoderAttention>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

This DecoderAttention supports self attention and cross attention, key and value cache, and key_padding_mask. The attention mask is not support at the moment.
Some boolean parameters are passed by runtime input for generic purpose

**Attributes**

* **num_heads** (required):
  Number of attention heads Default value is ``?``.

**Inputs**

* **query** (heterogeneous) - **T**:
  3D input tensor with shape (sequence_length, batch_size,
  hidden_size), hidden_size = num_heads * head_size
* **key** (heterogeneous) - **T**:
  3D input tensor with shape (total_sequence_length, batch_size,
  hidden_size)
* **q_weight** (heterogeneous) - **T**:
  2D input tensor with shape (hidden_size, hidden_size)
* **kv_weight** (heterogeneous) - **T**:
  2D input tensor with shape (hidden_size, 2 * hidden_size)
* **bias** (heterogeneous) - **T**:
  1D input tensor with shape (3 * hidden_size)
* **key_padding_mask** (optional, heterogeneous) - **B**:
  2D input tensor with shape (batch_size, total_sequence_length)
* **key_cache** (optional, heterogeneous) - **T**:
  input tensor with shape (batch_size, num_heads, sequence_length or
  total_sequence_length, head_size)
* **value_cache** (optional, heterogeneous) - **T**:
  input tensor with shape (batch_size, num_heads, sequence_length or
  total_sequence_length, head_size)
* **static_kv** (heterogeneous) - **B**:
  If static_kv = true, cross-attention; else self-attention
* **use_past** (heterogeneous) - **B**:
  If use_past = true, use cache; else no cache
* **has_layer_state** (heterogeneous) - **B**:
  If has_layer_state = true, layer_state = {} or [a,b]; else
  layer_state = None
* **has_key_padding_mask** (heterogeneous) - **B**:
  has_key_padding_mask or not

**Outputs**

Between 1 and 3 outputs.

* **output** (heterogeneous) - **T**:
  3D output tensor with shape (sequence_length, batch_size,
  hidden_size)
* **new_key_cache** (optional, heterogeneous) - **T**:
  output tensor with shape (batch_size, num_heads, new
  sequence_length, head_size)
* **new_value_cache** (optional, heterogeneous) - **T**:
  output tensor with shape (batch_size, num_heads, new
  sequence_length, head_size)

**Examples**
