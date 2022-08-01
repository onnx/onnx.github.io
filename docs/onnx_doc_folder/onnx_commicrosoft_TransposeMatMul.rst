
.. _l-onnx-doccom.microsoft-TransposeMatMul:

===============================
com.microsoft - TransposeMatMul
===============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-transposematmul-1:

TransposeMatMul - 1 (com.microsoft)
===================================

**Version**

* **name**: `TransposeMatMul (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.TransposeMatMul>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Duplicate of FusedMatMul. Going forward FusedMatMul should be used. This OP will be supported for backward compatibility.
Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html

**Attributes**

* **alpha**:
  Scalar multiplier for the product of the input tensors. Default value is ``?``.
* **transA**:
  Whether A should be transposed on the last two dimensions before
  doing multiplication Default value is ``?``.
* **transB**:
  Whether B should be transposed on the last two dimensions before
  doing multiplication Default value is ``?``.

**Inputs**

* **A** (heterogeneous) - **T**:
  N-dimensional matrix A
* **B** (heterogeneous) - **T**:
  N-dimensional matrix B

**Outputs**

* **Y** (heterogeneous) - **T**:
  Matrix multiply results

**Examples**
