
.. _l-onnx-doccom.microsoft-BiasGeluGrad_dX:

===============================
com.microsoft - BiasGeluGrad_dX
===============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-biasgelugrad_dx-1:

BiasGeluGrad_dX - 1 (com.microsoft)
===================================

**Version**

* **name**: `BiasGeluGrad_dX (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.BiasGeluGrad_dX>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Computes dX for BiasGeluGrad

**Inputs**

* **dY** (heterogeneous) - **T**:
  The gradient tensor from output.
* **X** (heterogeneous) - **T**:
  The input tensor.
* **B** (heterogeneous) - **T**:
  The bias tensor.

**Outputs**

* **dX** (heterogeneous) - **T**:
  Gradient of the input.

**Examples**
