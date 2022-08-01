
.. _l-onnx-doccom.microsoft-QLinearMul:

==========================
com.microsoft - QLinearMul
==========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-qlinearmul-1:

QLinearMul - 1 (com.microsoft)
==============================

**Version**

* **name**: `QLinearMul (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.QLinearMul>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Performs element-wise binary multiplication on 8 bit data types (with Numpy-style broadcasting support).

C = ((A - A_zero_point) * (B - B_zero_point)) * (A_scale * B_scale)/C_scale + C_zero_point

**Inputs**

Between 7 and 8 inputs.

* **A** (heterogeneous) - **T**:
  First operand.
* **A_scale** (heterogeneous) - **tensor(float)**:
  Input A's scale. It's a scalar, which means a per-tensor/layer
  quantization.
* **A_zero_point** (optional, heterogeneous) - **T**:
  Input A zero point. Default value is 0 if it's not specified. It's a
  scalar, which means a per-tensor/layer quantization.
* **B** (heterogeneous) - **T**:
  Second operand.
* **B_scale** (heterogeneous) - **tensor(float)**:
  Input B's scale. It's a scalar, which means a per-tensor/layer
  quantization.
* **B_zero_point** (optional, heterogeneous) - **T**:
  Input B zero point. Default value is 0 if it's not specified. It's a
  scalar, which means a per-tensor/layer quantization.
* **C_scale** (heterogeneous) - **tensor(float)**:
  Output scale. It's a scalar, which means a per-tensor/layer
  quantization.
* **C_zero_point** (optional, heterogeneous) - **T**:
  Output zero point. Default value is 0 if it's not specified. It's a
  scalar, which means a per-tensor/layer quantization.

**Outputs**

* **C** (heterogeneous) - **T**:
  Result, has same element type as two inputs

**Examples**
