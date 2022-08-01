
.. _l-onnx-doccom.microsoft-SkipLayerNormalization:

======================================
com.microsoft - SkipLayerNormalization
======================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-skiplayernormalization-1:

SkipLayerNormalization - 1 (com.microsoft)
==========================================

**Version**

* **name**: `SkipLayerNormalization (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.SkipLayerNormalization>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Skip and Layer Normalization Fusion

**Attributes**

* **epsilon**:
  The epsilon value to use to avoid division by zero. Default value is ``?``.

**Inputs**

Between 3 and 5 inputs.

* **input** (heterogeneous) - **T**:
  3D input tensor with shape (batch_size, sequence_length,
  hidden_size)
* **skip** (heterogeneous) - **T**:
  3D skip tensor with shape (batch_size, sequence_length, hidden_size)
* **gamma** (heterogeneous) - **T**:
  1D input tensor with shape (hidden_size)
* **beta** (optional, heterogeneous) - **T**:
  1D skip tensor with shape (hidden_size
* **bias** (optional, heterogeneous) - **T**:
  1D bias tensor with shape (hidden_size

**Outputs**

Between 1 and 3 outputs.

* **output** (heterogeneous) - **T**:
  3D output tensor with shape (batch_size, sequence_length,
  hidden_size)
* **mean** (optional, heterogeneous) - **U**:
  Saved mean used during training to speed up gradient computation
* **inv_std_var** (optional, heterogeneous) - **U**:
  Saved inverse standard variance used during training to speed up
  gradient computation.

**Examples**
