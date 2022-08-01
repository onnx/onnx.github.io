
.. _l-onnx-doccom.microsoft-YieldOp:

=======================
com.microsoft - YieldOp
=======================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-yieldop-1:

YieldOp - 1 (com.microsoft)
===========================

**Version**

* **name**: `YieldOp (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.YieldOp>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Yield Op.

**Attributes**

* **full_shape_outputs** (required):
  The indices of the module outputs that must have full shape. Default value is ``?``.
* **non_differentiable_outputs**:
  The indices of the module outputs that doesn't have a gradient. Default value is ``?``.

**Inputs**

Between 1 and 2147483647 inputs.

* **module_outputs** (variadic) - **T**:
  Module outputs to be returned to pytorch.

**Outputs**

Between 0 and 2147483647 outputs.

* **module_outputs_grad** (variadic) - **T**:
  Gradient of module outputs returned from pytorch.

**Examples**
