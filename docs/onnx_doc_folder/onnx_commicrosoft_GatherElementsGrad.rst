
.. _l-onnx-doccom.microsoft-GatherElementsGrad:

==================================
com.microsoft - GatherElementsGrad
==================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-gatherelementsgrad-1:

GatherElementsGrad - 1 (com.microsoft)
======================================

**Version**

* **name**: `GatherElementsGrad (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.GatherElementsGrad>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

GatherElementsGrad

**Attributes**

* **axis**:
  Which axis to scatter on. Negative value means counting dimensions
  from the back. Accepted range is [-r, r-1] where r = rank(data). Default value is ``?``.

**Inputs**

* **dY** (heterogeneous) - **T**:
  Tensor of rank r >=1 (same rank and shape as indices)
* **shape** (heterogeneous) - **I**:
  Shape of the GatherElements input data.
* **indices** (heterogeneous) - **Tind**:
  Tensor of int32/int64 indices, of r >= 1 (same rank as input). All
  index values are expected to be within bounds [-s, s-1] along axis
  of size s. It is an error if any of the index values are out of
  bounds.

**Outputs**

* **dX** (heterogeneous) - **T**:
  Tensor of rank r >= 1 (same rank as input).

**Examples**
