
.. _l-onnx-doccom.microsoft-MatMulIntegerToFloat:

====================================
com.microsoft - MatMulIntegerToFloat
====================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-matmulintegertofloat-1:

MatMulIntegerToFloat - 1 (com.microsoft)
========================================

**Version**

* **name**: `MatMulIntegerToFloat (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.MatMulIntegerToFloat>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

**Inputs**

Between 4 and 7 inputs.

* **A** (heterogeneous) - **T1**:
  N-dimensional matrix A
* **B** (heterogeneous) - **T2**:
  N-dimensional matrix B
* **a_scale** (heterogeneous) - **T3**:
  Scale of quantized input 'A'. It could be a scalar or a 1-D tensor,
  which means a per-tensor or per-column quantization. If it's a 1-D
  tensor, its number of elements should be equal to the number of
  columns of input 'A'.
* **b_scale** (heterogeneous) - **T3**:
  Scale of quantized input 'B'. It could be a scalar or a 1-D tensor,
  which means a per-tensor or per-column quantization. If it's a 1-D
  tensor, its number of elements should be equal to the number of
  columns of input 'B'.
* **a_zero_point** (optional, heterogeneous) - **T1**:
  Zero point tensor for input 'A'. It's optional and default value is
  0.  It could be a scalar or a 1-D tensor, which means a per-tensor
  or per-column quantization. If it's a 1-D tensor, its number of
  elements should be equal to the number of columns of input 'A'.
* **b_zero_point** (optional, heterogeneous) - **T2**:
  Zero point tensor for input 'B'. It's optional and default value is
  0.  It could be a scalar or a 1-D tensor, which means a per-tensor
  or per-column quantization. If it's a 1-D tensor, its number of
  elements should be equal to the number of columns of input 'B'.
* **bias** (optional, heterogeneous) - **T3**:
  1D input tensor, whose dimension is same as B's last dimension

**Outputs**

* **Y** (heterogeneous) - **T3**:
  Matrix multiply results from A * B

**Examples**
