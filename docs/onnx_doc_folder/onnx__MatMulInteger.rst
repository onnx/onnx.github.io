
.. _l-onnx-doc-MatMulInteger:

=============
MatMulInteger
=============

.. contents::
    :local:


.. _l-onnx-op-matmulinteger-10:

MatMulInteger - 10
==================

**Version**

* **name**: `MatMulInteger (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMulInteger>`_
* **domain**: **main**
* **since_version**: **10**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 10**.

**Summary**

Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.

**Inputs**

Between 2 and 4 inputs.

* **A** (heterogeneous) - **T1**:
  N-dimensional matrix A
* **B** (heterogeneous) - **T2**:
  N-dimensional matrix B
* **a_zero_point** (optional, heterogeneous) - **T1**:
  Zero point tensor for input 'A'. It's optional and default value is
  0. It could be a scalar or N-D tensor. Scalar refers to per tensor
  quantization whereas N-D refers to per row quantization. If the
  input is 2D of shape [M, K] then zero point tensor may be an M
  element vector [zp_1, zp_2, ..., zp_M]. If the input is N-D tensor
  with shape [D1, D2, M, K] then zero point tensor may have shape [D1,
  D2, M, 1].
* **b_zero_point** (optional, heterogeneous) - **T2**:
  Zero point tensor for input 'B'. It's optional and default value is
  0. It could be a scalar or a N-D tensor, Scalar refers to per tensor
  quantization whereas N-D refers to per col quantization. If the
  input is 2D of shape [K, N] then zero point tensor may be an N
  element vector [zp_1, zp_2, ..., zp_N]. If the input is N-D tensor
  with shape [D1, D2, K, N] then zero point tensor may have shape [D1,
  D2, 1, N].

**Outputs**

* **Y** (heterogeneous) - **T3**:
  Matrix multiply results from A * B

**Type Constraints**

* **T1** in (
  tensor(int8),
  tensor(uint8)
  ):
  Constrain input A data type to 8-bit integer tensor.
* **T2** in (
  tensor(int8),
  tensor(uint8)
  ):
  Constrain input B data type to 8-bit integer tensor.
* **T3** in (
  tensor(int32)
  ):
  Constrain output Y data type as 32-bit integer tensor.

**Examples**
