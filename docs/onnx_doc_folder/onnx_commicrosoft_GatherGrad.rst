
.. _l-onnx-doccom.microsoft-GatherGrad:

==========================
com.microsoft - GatherGrad
==========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-gathergrad-1:

GatherGrad - 1 (com.microsoft)
==============================

**Version**

* **name**: `GatherGrad (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.GatherGrad>`_
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
  Which axis to gather on. Negative value means counting dimensions
  from the back. Accepted range in [-r, r-1] Default value is ``?``.

**Inputs**

* **shape** (heterogeneous) - **I**:
  Shape of the Gather input X.
* **indices** (heterogeneous) - **Tind**:
  Tensor of int32/int64 indices, of any rank q.
* **dY** (heterogeneous) - **T**:
  Gradient of output

**Outputs**

* **dX** (heterogeneous) - **T**:
  Gradient of input

**Examples**
