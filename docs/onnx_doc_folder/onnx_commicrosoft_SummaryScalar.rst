
.. _l-onnx-doccom.microsoft-SummaryScalar:

=============================
com.microsoft - SummaryScalar
=============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-summaryscalar-1:

SummaryScalar - 1 (com.microsoft)
=================================

**Version**

* **name**: `SummaryScalar (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.SummaryScalar>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

SummaryScalar

**Attributes**

* **tags** (required):
  The tags corresponding to each input scalar. Default value is ``?``.

**Inputs**

* **input** (heterogeneous) - **T**:
  The scalar tensor to summarize as simple values.

**Outputs**

* **summary** (heterogeneous) - **S**:
  The serialized Tensorboard Summary.

**Examples**
