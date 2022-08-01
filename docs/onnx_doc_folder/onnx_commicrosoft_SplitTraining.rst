
.. _l-onnx-doccom.microsoft-SplitTraining:

=============================
com.microsoft - SplitTraining
=============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-splittraining-1:

SplitTraining - 1 (com.microsoft)
=================================

**Version**

* **name**: `SplitTraining (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.SplitTraining>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

SplitTraining

**Attributes**

* **axis**:
  Which axis to split on. A negative value means counting dimensions
  from the back. Accepted range is [-rank, rank-1] where r =
  rank(input). Default value is ``?``.

**Inputs**

* **input** (heterogeneous) - **T**:
  The tensor to split
* **split** (heterogeneous) - **tensor(int64)**:
  length of each output

**Outputs**

Between 1 and 2147483647 outputs.

* **outputs** (variadic, heterogeneous) - **T**:
  One or more outputs forming list of tensors after splitting

**Examples**
