
.. _l-onnx-doccom.microsoft-ConcatTraining:

==============================
com.microsoft - ConcatTraining
==============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-concattraining-1:

ConcatTraining - 1 (com.microsoft)
==================================

**Version**

* **name**: `ConcatTraining (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.ConcatTraining>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Concatenate a list of tensors into a single tensor

**Attributes**

* **axis** (required):
  Which axis to concat on Default value is ``?``.

**Inputs**

Between 1 and 2147483647 inputs.

* **inputs** (variadic, heterogeneous) - **T**:
  List of tensors for concatenation

**Outputs**

Between 1 and 2 outputs.

* **concat_result** (heterogeneous) - **T**:
  Concatenated tensor
* **per_input_length** (optional, heterogeneous) - **Tint**:
  Vector of length of each concatenated input along the 'axis'
  dimension

**Examples**
