
.. _l-onnx-doccom.microsoft-NGramRepeatBlock:

================================
com.microsoft - NGramRepeatBlock
================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-ngramrepeatblock-1:

NGramRepeatBlock - 1 (com.microsoft)
====================================

**Version**

* **name**: `NGramRepeatBlock (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.NGramRepeatBlock>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Enforce no repetition of n-grams. Scores are set to `-inf` for tokens that form a repeated n-gram if added to the back of the input_ids.

**Attributes**

* **ngram_size** (required):
  The NGram size. Default value is ``?``.

**Inputs**

* **input_ids** (heterogeneous) - **Tid**:
  2D input tensor with shape (batch_size, sequence_length)
* **scores** (heterogeneous) - **T**:
  2D input tensor with shape (batch_size, vocab_size)

**Outputs**

* **scores_out** (heterogeneous) - **T**:
  2D output tensor with shape (batch_size, vocab_size)

**Examples**
