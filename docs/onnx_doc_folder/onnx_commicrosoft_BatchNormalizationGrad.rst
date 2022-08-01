
.. _l-onnx-doccom.microsoft-BatchNormalizationGrad:

======================================
com.microsoft - BatchNormalizationGrad
======================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-batchnormalizationgrad-1:

BatchNormalizationGrad - 1 (com.microsoft)
==========================================

**Version**

* **name**: `BatchNormalizationGrad (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.BatchNormalizationGrad>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

BatchNormalizationGrad

**Attributes**

* **epsilon** (required):
  epsilon value Default value is ``?``.

**Inputs**

* **dY** (heterogeneous) - **T**:
  Gradient output from previous node
* **X** (heterogeneous) - **T**:
  Input
* **scale** (heterogeneous) - **T1**:
  Scale tensor
* **mean** (heterogeneous) - **T2**:
  Mean of X
* **variance** (heterogeneous) - **T2**:
  Variance of X

**Outputs**

* **X_grad** (heterogeneous) - **T**:
  Gradient of the input
* **scale_grad** (heterogeneous) - **T1**:
  Gradient of the scale
* **bias_grad** (heterogeneous) - **T1**:
  Gradient of the bias

**Examples**
