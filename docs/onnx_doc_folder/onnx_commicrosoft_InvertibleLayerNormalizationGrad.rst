
.. _l-onnx-doccom.microsoft-InvertibleLayerNormalizationGrad:

================================================
com.microsoft - InvertibleLayerNormalizationGrad
================================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-invertiblelayernormalizationgrad-1:

InvertibleLayerNormalizationGrad - 1 (com.microsoft)
====================================================

**Version**

* **name**: `InvertibleLayerNormalizationGrad (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.InvertibleLayerNormalizationGrad>`_
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
* **Y** (heterogeneous) - **V**:
  Output data tensor from the forward path
* **scale** (heterogeneous) - **V**:
  Scale tensor.
* **bias** (heterogeneous) - **V**:
  Bias tensor.
* **inv_std_var** (heterogeneous) - **U**:
  inverse std variance of X.

**Outputs**

* **X_grad** (heterogeneous) - **T**:
  Gradient of the input.
* **scale_grad** (heterogeneous) - **V**:
  Gradient of the scale.
* **bias_grad** (heterogeneous) - **V**:
  Gradient of the bias.

**Examples**
