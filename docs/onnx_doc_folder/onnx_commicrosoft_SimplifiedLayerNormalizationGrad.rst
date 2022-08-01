
.. _l-onnx-doccom.microsoft-SimplifiedLayerNormalizationGrad:

================================================
com.microsoft - SimplifiedLayerNormalizationGrad
================================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-simplifiedlayernormalizationgrad-1:

SimplifiedLayerNormalizationGrad - 1 (com.microsoft)
====================================================

**Version**

* **name**: `SimplifiedLayerNormalizationGrad (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.SimplifiedLayerNormalizationGrad>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

SimplifiedLayerNormalizationGrad

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
* **inv_std_var** (heterogeneous) - **U**:
  inverse std variance of X.

**Outputs**

* **X_grad** (heterogeneous) - **T**:
  Gradient of the input.
* **scale_grad** (heterogeneous) - **V**:
  Gradient of the scale.

**Examples**
