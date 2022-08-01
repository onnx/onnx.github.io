
.. _l-onnx-doccom.microsoft-SummaryText:

===========================
com.microsoft - SummaryText
===========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-summarytext-1:

SummaryText - 1 (com.microsoft)
===============================

**Version**

* **name**: `SummaryText (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.SummaryText>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

SummaryText

**Attributes**

* **tag** (required):
  The tag corresponding to the text data. Default value is ``?``.

**Inputs**

* **input** (heterogeneous) - **S**:
  The string tensor to render in the Tensorboard Text dashboard.

**Outputs**

* **summary** (heterogeneous) - **S**:
  The serialized Tensorboard Summary.

**Examples**
