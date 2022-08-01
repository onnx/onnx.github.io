
.. _l-onnx-doccom.microsoft-EmbedLayerNormalization:

=======================================
com.microsoft - EmbedLayerNormalization
=======================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-embedlayernormalization-1:

EmbedLayerNormalization - 1 (com.microsoft)
===========================================

**Version**

* **name**: `EmbedLayerNormalization (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.EmbedLayerNormalization>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

EmbedLayerNormalization is the fusion of embedding layer in BERT model, with optional mask processing.
The embedding layer takes input_ids (word IDs) and segment_ids (sentence IDs) to look up word_embedding, position_embedding,
and segment_emedding; the embeddings are added then applied layer normalization using gamma and beta tensors.
The last input mask is optional. If mask is provided, mask index (that is position of first 0 in mask, or number of words)
will be calculated.

**Attributes**

* **epsilon**:
  The epsilon value to use to avoid division by zero. Default value is ``?``.

**Inputs**

Between 7 and 9 inputs.

* **input_ids** (heterogeneous) - **T1**:
  2D words IDs with shape (batch_size, sequence_length)
* **segment_ids** (optional, heterogeneous) - **T1**:
  2D segment IDs with shape (batch_size, sequence_length)
* **word_embedding** (heterogeneous) - **T**:
  2D with shape (,hidden_size)
* **position_embedding** (heterogeneous) - **T**:
  2D with shape (, hidden_size)
* **segment_embedding** (optional, heterogeneous) - **T**:
  2D with shape (, hidden_size)
* **gamma** (heterogeneous) - **T**:
  1D gamma tensor for layer normalization with shape (hidden_size)
* **beta** (heterogeneous) - **T**:
  1D beta tensor for layer normalization  with shape (hidden_size)
* **mask** (optional, heterogeneous) - **T1**:
  2D attention mask with shape (batch_size, sequence_length)
* **position_ids** (optional, heterogeneous) - **T1**:
  2D position ids with shape (batch_size, sequence_length)

**Outputs**

Between 2 and 3 outputs.

* **output** (heterogeneous) - **T**:
  3D output tensor with shape (batch_size, sequence_length,
  hidden_size)
* **mask_index** (heterogeneous) - **T1**:
  1D mask_index tensor with shape (batch_size)
* **embedding_sum** (optional, heterogeneous) - **T**:
  sum of word_embedding and position_embedding without layer
  normalization

**Examples**
