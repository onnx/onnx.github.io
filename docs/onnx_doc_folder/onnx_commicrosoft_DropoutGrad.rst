
.. _l-onnx-doccom.microsoft-DropoutGrad:

===========================
com.microsoft - DropoutGrad
===========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-dropoutgrad-1:

DropoutGrad - 1 (com.microsoft)
===============================

**Version**

* **name**: `DropoutGrad (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.DropoutGrad>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

DropoutGrad

**Inputs**

Between 2 and 4 inputs.

* **dy** (heterogeneous) - **T**:
  The gradient tensor from output.
* **mask** (heterogeneous) - **T2**:
  The mask output of the dropout.
* **ratio** (optional, heterogeneous) - **T1**:
  Same value as the ratio input supplied to the dropout op with value
  in [0, 1). If this input is not specified, a default value of 0.5 is
  used.
* **training_mode** (optional, heterogeneous) - **T2**:
  Same value as the training_mode input supplied to the dropout op. If
  this input is not specified, a default value of false is used.

**Outputs**

* **dx** (heterogeneous) - **T**:
  Gradient of the input.

**Examples**
