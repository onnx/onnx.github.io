
.. _l-onnx-doccom.microsoft-Inverse:

=======================
com.microsoft - Inverse
=======================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-inverse-1:

Inverse - 1 (com.microsoft)
===========================

**Version**

* **name**: `Inverse (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.Inverse>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

**Inputs**

* **X** (heterogeneous) - **T**:
  Input tensor. Every matrix in the batch must be invertible.

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output tensor of the same type and shape as the input tensor.

**Examples**
