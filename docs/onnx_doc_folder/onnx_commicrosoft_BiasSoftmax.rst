
.. _l-onnx-doccom.microsoft-BiasSoftmax:

===========================
com.microsoft - BiasSoftmax
===========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-biassoftmax-1:

BiasSoftmax - 1 (com.microsoft)
===============================

**Version**

* **name**: `BiasSoftmax (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.BiasSoftmax>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Y = softmax(scores + bias)) with simple broadcast on bias. Intended to specialize softmax(scores + additive_mask) commonly found in transformer models.

**Attributes**

* **broadcast_axis**:
  broadcast bias across input for dimensions broadcast_axis to
  softmax_axis-1 Default value is ``?``.
* **softmax_axis**:
  apply softmax to elements for dimensions softmax_axis or higher Default value is ``?``.

**Inputs**

* **data** (heterogeneous) - **T**:
  The input data as Tensor.
* **bias** (heterogeneous) - **T**:
  The bias (or mask) as Tensor.

**Outputs**

* **output** (heterogeneous) - **T**:
  The output.

**Examples**
