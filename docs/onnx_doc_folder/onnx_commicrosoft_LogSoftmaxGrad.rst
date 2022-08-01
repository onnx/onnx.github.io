
.. _l-onnx-doccom.microsoft-LogSoftmaxGrad:

==============================
com.microsoft - LogSoftmaxGrad
==============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-logsoftmaxgrad-1:

LogSoftmaxGrad - 1 (com.microsoft)
==================================

**Version**

* **name**: `LogSoftmaxGrad (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.LogSoftmaxGrad>`_
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
  Describes the axis of the inputs when coerced to 2D; defaults to one
  because the 0th axis most likely describes the batch_size Default value is ``?``.

**Inputs**

* **dY** (heterogeneous) - **T**:
  Gradient of output Y
* **X** (heterogeneous) - **T**:
  Input tensor

**Outputs**

* **dX** (heterogeneous) - **T**:
  Gradient of input X

**Examples**
