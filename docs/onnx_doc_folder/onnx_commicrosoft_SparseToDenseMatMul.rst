
.. _l-onnx-doccom.microsoft-SparseToDenseMatMul:

===================================
com.microsoft - SparseToDenseMatMul
===================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-sparsetodensematmul-1:

SparseToDenseMatMul - 1 (com.microsoft)
=======================================

**Version**

* **name**: `SparseToDenseMatMul (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.SparseToDenseMatMul>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

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
  2-dimensional sparse matrix A. Either COO or CSR format
* **B** (heterogeneous) - **T1**:
  N-dimensional dense matrix B

**Outputs**

* **Y** (heterogeneous) - **T1**:
  Matrix multiply results

**Examples**
