
.. _l-onnx-doccom.microsoft-WordConvEmbedding:

=================================
com.microsoft - WordConvEmbedding
=================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-wordconvembedding-1:

WordConvEmbedding - 1 (com.microsoft)
=====================================

**Version**

* **name**: `WordConvEmbedding (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.WordConvEmbedding>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

The WordConvEmbedding takes in a batch of sequence words and embed each word to a vector.

**Attributes**

* **char_embedding_size**:
  Integer representing the embedding vector size for each char.If not
  provide, use the char embedding size of embedding vector. Default value is ``?``.
* **conv_window_size**:
  This operator applies convolution to word from left to right with
  window equal to conv_window_size and stride to 1.Take word 'example'
  for example, with conv_window_size equal to 2, conv is applied to
  [ex],[xa], [am], [mp]...If not provide, use the first dimension of
  conv kernal shape. Default value is ``?``.
* **embedding_size**:
  Integer representing the embedding vector size for each word.If not
  provide, use the fileter size of conv weight Default value is ``?``.

**Inputs**

* **Sequence** (heterogeneous) - **T**:
  Specify batchs of sequence words to embedding
* **W** (heterogeneous) - **T1**:
  Specify weights of conv
* **B** (heterogeneous) - **T1**:
  Specify bias of conv
* **C** (heterogeneous) - **T1**:
  Specify embedding vector of char

**Outputs**

* **Y** (heterogeneous) - **T1**:
  output

**Examples**
