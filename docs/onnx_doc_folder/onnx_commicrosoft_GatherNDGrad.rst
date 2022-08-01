
.. _l-onnx-doccom.microsoft-GatherNDGrad:

============================
com.microsoft - GatherNDGrad
============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-gatherndgrad-1:

GatherNDGrad - 1 (com.microsoft)
================================

**Version**

* **name**: `GatherNDGrad (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.GatherNDGrad>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

**Attributes**

* **batch_dims**:
  The number of batch dims. The gather of indexing starts from
  dimension of data[batch_dims+1:] Default value is ``?``.

**Inputs**

* **shape** (heterogeneous) - **T1**:
  The shape of source data input of GatherND.
* **indices** (heterogeneous) - **Tind**:
  Tensor of rank q >= 1.
* **update** (heterogeneous) - **T**:
  The gradient of the output.

**Outputs**

* **output** (heterogeneous) - **T**:
  Tensor graident of the input.

**Examples**
