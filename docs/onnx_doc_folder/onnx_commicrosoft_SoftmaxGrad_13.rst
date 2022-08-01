
.. _l-onnx-doccom.microsoft-SoftmaxGrad_13:

==============================
com.microsoft - SoftmaxGrad_13
==============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-softmaxgrad_13-1:

SoftmaxGrad_13 - 1 (com.microsoft)
==================================

**Version**

* **name**: `SoftmaxGrad_13 (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.SoftmaxGrad_13>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

**Attributes**

* **axis**:
  Describes the dimension Softmax will be performed on.Defaults to -1.
  Negative value means counting dimensions from the back. Default value is ``?``.

**Inputs**

* **dY** (heterogeneous) - **T**:
  Gradient of output Y
* **Y** (heterogeneous) - **T**:
  Input tensor

**Outputs**

* **dX** (heterogeneous) - **T**:
  Gradient of input X

**Examples**
