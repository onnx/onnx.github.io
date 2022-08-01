
.. _l-onnx-doccom.microsoft-FusedMatMul:

===========================
com.microsoft - FusedMatMul
===========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-fusedmatmul-1:

FusedMatMul - 1 (com.microsoft)
===============================

**Version**

* **name**: `FusedMatMul (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.FusedMatMul>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

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
* **transBatchA**:
  Whether A should be transposed on the 1st dimension and batch
  dimensions (dim-1 to dim-rank-2) before doing multiplication Default value is ``?``.
* **transBatchB**:
  Whether B should be transposed on the 1st dimension and batch
  dimensions (dim-1 to dim-rank-2) before doing multiplication Default value is ``?``.

**Inputs**

* **A** (heterogeneous) - **T**:
  N-dimensional matrix A
* **B** (heterogeneous) - **T**:
  N-dimensional matrix B

**Outputs**

* **Y** (heterogeneous) - **T**:
  Matrix multiply results

**Examples**
