
.. _l-onnx-doccom.microsoft-MatMulInteger16:

===============================
com.microsoft - MatMulInteger16
===============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-matmulinteger16-1:

MatMulInteger16 - 1 (com.microsoft)
===================================

**Version**

* **name**: `MatMulInteger16 (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.MatMulInteger16>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
 The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.

**Inputs**

* **A** (heterogeneous) - **T1**:
  N-dimensional matrix A
* **B** (heterogeneous) - **T2**:
  N-dimensional matrix B

**Outputs**

* **Y** (heterogeneous) - **T3**:
  Matrix multiply results from A * B

**Examples**
