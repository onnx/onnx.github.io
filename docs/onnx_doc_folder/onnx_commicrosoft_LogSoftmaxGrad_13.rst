
.. _l-onnx-doccom.microsoft-LogSoftmaxGrad_13:

=================================
com.microsoft - LogSoftmaxGrad_13
=================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-logsoftmaxgrad_13-1:

LogSoftmaxGrad_13 - 1 (com.microsoft)
=====================================

**Version**

* **name**: `LogSoftmaxGrad_13 (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.LogSoftmaxGrad_13>`_
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
  Describes the dimension LogSoftmax will be performed on.Defaults to
  -1. Negative value means counting dimensions from the back. Default value is ``?``.

**Inputs**

* **dY** (heterogeneous) - **T**:
  Gradient of output Y
* **X** (heterogeneous) - **T**:
  Input tensor

**Outputs**

* **dX** (heterogeneous) - **T**:
  Gradient of input X

**Examples**
