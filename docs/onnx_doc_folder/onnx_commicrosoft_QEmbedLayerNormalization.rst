
.. _l-onnx-doccom.microsoft-QEmbedLayerNormalization:

========================================
com.microsoft - QEmbedLayerNormalization
========================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-qembedlayernormalization-1:

QEmbedLayerNormalization - 1 (com.microsoft)
============================================

**Version**

* **name**: `QEmbedLayerNormalization (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.QEmbedLayerNormalization>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

QEmbedLayerNormalization is the quantized fusion of embedding layer in BERT model, with optional mask processing.
The embedding layer takes input_ids (word IDs) and segment_ids (sentence IDs) to look up word_embedding, position_embedding,
and segment_emedding; the embeddings are added then applied layer normalization using gamma and beta tensors. The input_ids
and segment_ids remain int32. All embeddings, gamma, and beta tensors are converted to int8/uint8. The last input mask is optional.
If mask is provided, mask index (that is position of first 0 in mask, or number of words will be calculated.

**Attributes**

* **epsilon**:
  The epsilon value to use to avoid division by zero. Default value is ``?``.

**Inputs**

* **input_ids** (heterogeneous) - **T1**:
  2D words IDs with shape (batch_size, sequence_length)
* **segment_ids** (optional, heterogeneous) - **T1**:
  2D segment IDs with shape (batch_size, sequence_length)
* **word_embedding_quant** (heterogeneous) - **T2**:
  2D with shape (,hidden_size)
* **position_embedding_quant** (heterogeneous) - **T2**:
  2D with shape (, hidden_size)
* **segment_embedding** (optional, heterogeneous) - **T2**:
  2D with shape (, hidden_size)
* **gamma_quant** (heterogeneous) - **T2**:
  1D gamma tensor for layer normalization with shape (hidden_size)
* **beta_quant** (heterogeneous) - **T2**:
  1D beta tensor for layer normalization  with shape (hidden_size)
* **mask** (optional, heterogeneous) - **T1**:
  Mask
* **word_embedding_scale** (heterogeneous) - **T**:
  Scale for word embeddings
* **position_embedding_scale** (heterogeneous) - **T**:
  Scale for position embeddings
* **segment_embedding_scale** (optional, heterogeneous) - **T**:
  Scale for segment embeddings
* **gamma_scale** (heterogeneous) - **T**:
  Scale for 1D gamma tensor
* **beta_scale** (heterogeneous) - **T**:
  Scale for 1D beta tensor
* **word_embedding_zero_point** (heterogeneous) - **T2**:
  Zero point for word embeddings
* **position_embedding_zero_point** (heterogeneous) - **T2**:
  Zero point for position embeddings
* **segment_embedding_zero_point** (optional, heterogeneous) - **T2**:
  Zero Point for segment embeddings
* **gamma_zero_point** (heterogeneous) - **T2**:
  Zero Point for 1D gamma tensor
* **beta_zero_point** (heterogeneous) - **T2**:
  Zero Point for 1D beta tensor

**Outputs**

* **layernorm_out** (heterogeneous) - **T**:
  LayerNorm Output
* **mask_index_out** (heterogeneous) - **T1**:
  Mask Index Output

**Examples**
