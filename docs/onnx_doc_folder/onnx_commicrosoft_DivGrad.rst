
.. _l-onnx-doccom.microsoft-DivGrad:

=======================
com.microsoft - DivGrad
=======================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-divgrad-1:

DivGrad - 1 (com.microsoft)
===========================

**Version**

* **name**: `DivGrad (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.DivGrad>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

**Inputs**

* **dY** (heterogeneous) - **T**:
  Gradient of output
* **A** (heterogeneous) - **T**:
  dividend
* **B** (heterogeneous) - **T**:
  divisor

**Outputs**

Between 0 and 2 outputs.

* **dA** (optional, heterogeneous) - **T**:
  Gradient of dividend
* **dB** (optional, heterogeneous) - **T**:
  Gradient of divisor

**Examples**
