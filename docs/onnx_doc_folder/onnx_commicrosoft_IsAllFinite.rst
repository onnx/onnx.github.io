
.. _l-onnx-doccom.microsoft-IsAllFinite:

===========================
com.microsoft - IsAllFinite
===========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-isallfinite-1:

IsAllFinite - 1 (com.microsoft)
===============================

**Version**

* **name**: `IsAllFinite (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.IsAllFinite>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

IsAllFinite

**Attributes**

* **isinf_only**:
  If true, check only for Inf, -Inf. Default value is ``?``.
* **isnan_only**:
  If true, check only for NaN. Default value is ``?``.

**Inputs**

Between 1 and 2147483647 inputs.

* **input** (variadic, heterogeneous) - **V**:
  Input tensors to check.

**Outputs**

* **output** (heterogeneous) - **T**:
  The output scalar. Its value is true if all input tensors are
  finite. Otherwise, the output value would be false.

**Examples**
