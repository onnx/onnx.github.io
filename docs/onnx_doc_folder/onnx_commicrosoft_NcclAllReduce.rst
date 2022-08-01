
.. _l-onnx-doccom.microsoft-NcclAllReduce:

=============================
com.microsoft - NcclAllReduce
=============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-ncclallreduce-1:

NcclAllReduce - 1 (com.microsoft)
=================================

**Version**

* **name**: `NcclAllReduce (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.NcclAllReduce>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

**Attributes**

* **group_type**:
  0 - global parallel group, 1 - data parallel group, 2 - node local
  data parallel group, 3 - cross node data parallel group, 4 -
  horozontal parallel, 5 - model parallel. Default value is ``?``.

**Inputs**

Between 1 and 2147483647 inputs.

* **input** (variadic, heterogeneous) - **T**:
  tensors to be reduced

**Outputs**

Between 1 and 2147483647 outputs.

* **output** (variadic, heterogeneous) - **T**:
  reduced tensors

**Examples**
