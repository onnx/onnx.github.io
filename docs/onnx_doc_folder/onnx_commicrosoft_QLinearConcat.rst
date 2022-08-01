
.. _l-onnx-doccom.microsoft-QLinearConcat:

=============================
com.microsoft - QLinearConcat
=============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-qlinearconcat-1:

QLinearConcat - 1 (com.microsoft)
=================================

**Version**

* **name**: `QLinearConcat (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.QLinearConcat>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Concatenate a list of tensors into a single tensor.All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.

**Attributes**

* **axis** (required):
  Which axis to concat on Default value is ``?``.

**Inputs**

Between 3 and 2147483647 inputs.

* **Y_scale** (heterogeneous) - **TF**:
  Y's scale.
* **Y_zero_point** (heterogeneous) - **T8**:
  Y's zero point.
* **inputs** (variadic) - **TV**:
  List of tensors/scale/zero_point for concatenation

**Outputs**

* **Y** (heterogeneous) - **T8**:
  Concatenated tensor

**Examples**
