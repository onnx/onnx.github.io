
.. _l-onnx-doccom.microsoft-Range:

=====================
com.microsoft - Range
=====================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-range-1:

Range - 1 (com.microsoft)
=========================

**Version**

* **name**: `Range (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.Range>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Creates a sequence of numbers that begins at `start` and extends by increments of `delta`
up to but not including `limit`.

**Inputs**

Between 2 and 3 inputs.

* **start** (heterogeneous) - **T**:
  Tensor(scalar, or dims=[1]). First entry in the range.
* **limit** (heterogeneous) - **T**:
  Tensor(scalar, or dims=[1]). Upper limit of sequence, exclusive.
* **delta** (optional, heterogeneous) - **T**:
  Tensor(scalar, or dims=[1]). Number that increments start. Defaults
  to 1.

**Outputs**

* **Y** (heterogeneous) - **T**:
  1-D Tensor of the range.

**Examples**
