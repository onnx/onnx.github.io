
.. _l-onnx-doccom.microsoft-SoftmaxGrad:

===========================
com.microsoft - SoftmaxGrad
===========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-softmaxgrad-1:

SoftmaxGrad - 1 (com.microsoft)
===============================

**Version**

* **name**: `SoftmaxGrad (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.SoftmaxGrad>`_
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
  Describes the axis of the inputs when coerced to 2D; defaults to one
  because the 0th axis most likely describes the batch_size Default value is ``?``.

**Inputs**

* **dY** (heterogeneous) - **T**:
  Gradient of output Y
* **Y** (heterogeneous) - **T**:
  Input tensor

**Outputs**

* **dX** (heterogeneous) - **T**:
  Gradient of input X

**Examples**
