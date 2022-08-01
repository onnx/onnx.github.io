
.. _l-onnx-doccom.microsoft-ZeroGradient:

============================
com.microsoft - ZeroGradient
============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-zerogradient-1:

ZeroGradient - 1 (com.microsoft)
================================

**Version**

* **name**: `ZeroGradient (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.ZeroGradient>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

reset the accumulator for gradient

**Inputs**

* **old_gradient** (heterogeneous) - **T1**:
  historical result of accumulated gradient
* **reset_signal** (heterogeneous) - **T2**:
  if this input is available, it is ready to reset the accumulator

**Outputs**

* **zero_gradient** (heterogeneous) - **T1**:
  reset the gradient

**Examples**
