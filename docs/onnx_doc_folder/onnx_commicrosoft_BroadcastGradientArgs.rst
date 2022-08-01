
.. _l-onnx-doccom.microsoft-BroadcastGradientArgs:

=====================================
com.microsoft - BroadcastGradientArgs
=====================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-broadcastgradientargs-1:

BroadcastGradientArgs - 1 (com.microsoft)
=========================================

**Version**

* **name**: `BroadcastGradientArgs (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.BroadcastGradientArgs>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Returns the reduction axes for computing gradients of s0 op s1 with broadcast.The ouput axes are deterministic from last to first. Output is an empty vector when no reduction is necessary for the corresponding input.

**Inputs**

* **a_shape** (heterogeneous) - **T**:
  The 1st input shape as Tensor.
* **b_shape** (heterogeneous) - **T**:
  The 2nd input shape as Tensor.

**Outputs**

Between 0 and 2 outputs.

* **a_axes** (optional, heterogeneous) - **T**:
  The reduction axes for 1st input, last to first.
* **b_axes** (optional, heterogeneous) - **T**:
  The reduction axes for 2nd input, last to first.

**Examples**
