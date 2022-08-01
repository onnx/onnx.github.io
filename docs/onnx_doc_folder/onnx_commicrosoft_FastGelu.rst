
.. _l-onnx-doccom.microsoft-FastGelu:

========================
com.microsoft - FastGelu
========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-fastgelu-1:

FastGelu - 1 (com.microsoft)
============================

**Version**

* **name**: `FastGelu (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.FastGelu>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

GELU (Gaussian Error Linear Unit) approximation: Y=0.5*X*(1+tanh(0.797885*X+0.035677*X*X*X)) with an optional input of bias that will be added to X before GELU.

**Inputs**

Between 1 and 2 inputs.

* **X** (heterogeneous) - **T**:
  input tensor
* **bias** (optional, heterogeneous) - **T**:
  bias tensor

**Outputs**

* **Y** (heterogeneous) - **T**:
  output tensor

**Examples**
