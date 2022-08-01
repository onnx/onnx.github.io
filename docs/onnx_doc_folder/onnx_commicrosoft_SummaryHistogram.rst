
.. _l-onnx-doccom.microsoft-SummaryHistogram:

================================
com.microsoft - SummaryHistogram
================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-summaryhistogram-1:

SummaryHistogram - 1 (com.microsoft)
====================================

**Version**

* **name**: `SummaryHistogram (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.SummaryHistogram>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

SummaryHistogram

**Attributes**

* **tag** (required):
  The tag corresponding to the histogram data. Default value is ``?``.

**Inputs**

* **input** (heterogeneous) - **T**:
  The scalar tensor to produce a histogram over.

**Outputs**

* **summary** (heterogeneous) - **S**:
  The serialized Tensorboard Summary.

**Examples**
