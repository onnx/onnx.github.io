
.. _l-onnx-doccom.microsoft-AdasumAllReduce:

===============================
com.microsoft - AdasumAllReduce
===============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-adasumallreduce-1:

AdasumAllReduce - 1 (com.microsoft)
===================================

**Version**

* **name**: `AdasumAllReduce (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.AdasumAllReduce>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

**Attributes**

* **reduce_algo**:
  Algorithms for Adasum. Valid values are: CpuReduction(1) or
  GpuHierarchicalReduction(2) Default value is ``?``.

**Inputs**

Between 1 and 2147483647 inputs.

* **input** (variadic, heterogeneous) - **T**:
  tensors to be reduced

**Outputs**

Between 1 and 2147483647 outputs.

* **output** (variadic, heterogeneous) - **T**:
  reduced tensors

**Examples**
