
.. _l-onnx-doccom.microsoft-SummaryMerge:

============================
com.microsoft - SummaryMerge
============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-summarymerge-1:

SummaryMerge - 1 (com.microsoft)
================================

**Version**

* **name**: `SummaryMerge (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.SummaryMerge>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

SummaryMerge

**Inputs**

Between 1 and 2147483647 inputs.

* **input** (variadic, heterogeneous) - **S**:
  One or more serialized Tensorboard Summary tensors to merge into a
  single Summary.

**Outputs**

* **summary** (heterogeneous) - **S**:
  The serialized Tensorboard Summary.

**Examples**
