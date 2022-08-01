
.. _l-onnx-doccom.microsoft-LayerNormalizationGrad:

======================================
com.microsoft - LayerNormalizationGrad
======================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-layernormalizationgrad-1:

LayerNormalizationGrad - 1 (com.microsoft)
==========================================

**Version**

* **name**: `LayerNormalizationGrad (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.LayerNormalizationGrad>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

LayerNormalizationGrad

**Attributes**

* **axis**:
  The first normalization dimension: normalization will be performed
  along dimensions axis : rank(inputs). Default value is ``?``.

**Inputs**

* **Y_grad** (heterogeneous) - **V**:
  The gradient tensor from output.
* **X** (heterogeneous) - **T**:
  Input data tensor from the forward path
* **scale** (heterogeneous) - **V**:
  Scale tensor.
* **mean** (heterogeneous) - **U**:
  mean of X.
* **inv_std_dev** (heterogeneous) - **U**:
  inverse std deviation of X.

**Outputs**

* **X_grad** (heterogeneous) - **T**:
  Gradient of the input.
* **scale_grad** (heterogeneous) - **V**:
  Gradient of the scale.
* **bias_grad** (heterogeneous) - **V**:
  Gradient of the bias.

**Examples**
