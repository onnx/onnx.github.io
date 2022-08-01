
.. _l-onnx-doc-QLinearMatMul:

=============
QLinearMatMul
=============

.. contents::
    :local:


.. _l-onnx-op-qlinearmatmul-10:

QLinearMatMul - 10
==================

**Version**

* **name**: `QLinearMatMul (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#QLinearMatMul>`_
* **domain**: **main**
* **since_version**: **10**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 10**.

**Summary**

Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
It consumes two quantized input tensors, their scales and zero points, scale and zero point of output,
and computes the quantized output. The quantization formula is y = saturate((x / y_scale) + y_zero_point).
For (x / y_scale), it is rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
Scale and zero point must have same shape. They must be either scalar (per tensor) or N-D tensor
(per row for 'a' and per column for 'b'). Scalar refers to per tensor quantization whereas N-D refers to per row
or per column quantization. If the input is 2D of shape [M, K] then zero point and scale tensor may be
an M element vector [v_1, v_2, ..., v_M] for per row quantization and K element vector of shape [v_1, v_2, ..., v_K]
for per column quantization. If the input is N-D tensor with shape [D1, D2, M, K] then zero point and scale tensor may
have shape [D1, D2, M, 1] for per row quantization and shape [D1, D2, 1, K] for per column quantization.
Production must never overflow, and accumulation may overflow if and only if in 32 bits.

**Inputs**

* **a** (heterogeneous) - **T1**:
  N-dimensional quantized matrix a
* **a_scale** (heterogeneous) - **tensor(float)**:
  scale of quantized input a
* **a_zero_point** (heterogeneous) - **T1**:
  zero point of quantized input a
* **b** (heterogeneous) - **T2**:
  N-dimensional quantized matrix b
* **b_scale** (heterogeneous) - **tensor(float)**:
  scale of quantized input b
* **b_zero_point** (heterogeneous) - **T2**:
  zero point of quantized input b
* **y_scale** (heterogeneous) - **tensor(float)**:
  scale of quantized output y
* **y_zero_point** (heterogeneous) - **T3**:
  zero point of quantized output y

**Outputs**

* **y** (heterogeneous) - **T3**:
  Quantized matrix multiply results from a * b

**Type Constraints**

* **T1** in (
  tensor(int8),
  tensor(uint8)
  ):
  Constrain input a and its zero point data type to 8-bit integer
  tensor.
* **T2** in (
  tensor(int8),
  tensor(uint8)
  ):
  Constrain input b and its zero point data type to 8-bit integer
  tensor.
* **T3** in (
  tensor(int8),
  tensor(uint8)
  ):
  Constrain output y and its zero point data type to 8-bit integer
  tensor.

**Examples**
