
.. _l-onnx-doccom.microsoft-BatchNormInternal:

=================================
com.microsoft - BatchNormInternal
=================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-batchnorminternal-1:

BatchNormInternal - 1 (com.microsoft)
=====================================

**Version**

* **name**: `BatchNormInternal (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.BatchNormInternal>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Variant of BatchNormalization with additional output for saved_mean/inv_std_dev.

**Attributes**

* **epsilon**:
  epsilon value Default value is ``?``.
* **momentum**:
  momentum value Default value is ``?``.
* **training_mode**:
  true if training Default value is ``?``.

**Inputs**

* **X** (heterogeneous) - **T**:
  Input tensor.
* **scale** (heterogeneous) - **T1**:
  Scale tensor of shape (C).
* **B** (heterogeneous) - **T1**:
  Bias tensor of shape (C).
* **input_mean** (heterogeneous) - **T2**:
  running mean tensor of shape (C).
* **input_var** (heterogeneous) - **T2**:
  running variance tensor of shape (C).

**Outputs**

Between 1 and 5 outputs.

* **Y** (heterogeneous) - **T**:
  The output tensor of the same shape as X
* **running_mean** (optional, heterogeneous) - **T2**:
  The running mean after BN.
* **running_var** (optional, heterogeneous) - **T2**:
  Running var after BN
* **saved_mean** (optional, heterogeneous) - **T2**:
  Mean of the batch
* **saved_inv_std** (optional, heterogeneous) - **T2**:
  Inverse standard deviation for the batch

**Examples**
