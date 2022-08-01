
.. _l-onnx-doccom.microsoft-TorchEmbedding:

==============================
com.microsoft - TorchEmbedding
==============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-torchembedding-1:

TorchEmbedding - 1 (com.microsoft)
==================================

**Version**

* **name**: `TorchEmbedding (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.TorchEmbedding>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Based on Torch operator Embedding, creates a lookup table of embedding vectors of fixed size,
 for a dictionary of fixed size.

**Inputs**

Between 2 and 4 inputs.

* **weight** (heterogeneous) - **T**:
  The embedding matrix of size N x M. 'N' is equal to the maximum
  possible index + 1, and 'M' is equal to the embedding size
* **indices** (heterogeneous) - **tensor(int64)**:
  Long tensor containing the indices to extract from embedding matrix.
* **padding_idx** (optional, heterogeneous) - **tensor(int64)**:
  A 0-D scalar tensor. If specified, the entries at `padding_idx` do
  not contribute to the gradient; therefore, the embedding vector at
  `padding_idx` is not updated during training, i.e. it remains as a
  fixed pad.
* **scale_grad_by_freq** (optional, heterogeneous) - **tensor(bool)**:
  A 0-D bool tensor. If given, this will scale gradients by the
  inverse of frequency of the indices (words) in the mini-batch.
  Default  is ``False``

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output tensor of the same type as the input tensor. Shape of the
  output is * x M, where '*' is the shape of input indices, and 'M' is
  the embedding size.

**Examples**
