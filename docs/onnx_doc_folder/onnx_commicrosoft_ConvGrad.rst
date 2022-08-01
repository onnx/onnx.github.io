
.. _l-onnx-doccom.microsoft-ConvGrad:

========================
com.microsoft - ConvGrad
========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-convgrad-1:

ConvGrad - 1 (com.microsoft)
============================

**Version**

* **name**: `ConvGrad (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.ConvGrad>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

**Inputs**

* **dY** (heterogeneous) - **T**:
  Gradient of output Y
* **X** (heterogeneous) - **T**:
  Input tensor
* **W** (heterogeneous) - **T**:
  Weight tensor

**Outputs**

Between 0 and 3 outputs.

* **dX** (optional, heterogeneous) - **T**:
  Gradient of X
* **dW** (optional, heterogeneous) - **T**:
  Gradient of W
* **dB** (optional, heterogeneous) - **T**:
  Gradient of B

**Examples**
