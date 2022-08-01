
.. _l-onnx-doccom.microsoft-QGemm:

=====================
com.microsoft - QGemm
=====================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-qgemm-1:

QGemm - 1 (com.microsoft)
=========================

**Version**

* **name**: `QGemm (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.QGemm>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Quantized Gemm

**Attributes**

* **alpha**:
  Scalar multiplier for the product of input tensors A * B. Default value is ``?``.
* **transA**:
  Whether A should be transposed Default value is ``?``.
* **transB**:
  Whether B should be transposed Default value is ``?``.

**Inputs**

Between 6 and 9 inputs.

* **A** (heterogeneous) - **TA**:
  Input tensor A. The shape of A should be (M, K) if transA is 0, or
  (K, M) if transA is non-zero.
* **a_scale** (heterogeneous) - **T**:
  Scale of quantized input 'A'. It is a scalar,which means a per-
  tensor quantization.
* **a_zero_point** (heterogeneous) - **TA**:
  Zero point tensor for input 'A'. It is a scalar.
* **B** (heterogeneous) - **TB**:
  Input tensor B. The shape of B should be (K, N) if transB is 0, or
  (N, K) if transB is non-zero.
* **b_scale** (heterogeneous) - **T**:
  Scale of quantized input 'B'. It could be a scalar or a 1-D tensor,
  which means a per-tensor or per-column quantization. If it's a 1-D
  tensor, its number of elements should be equal to the number of
  columns of input 'B'.
* **b_zero_point** (heterogeneous) - **TB**:
  Zero point tensor for input 'B'. It's optional and default value is
  0.  It could be a scalar or a 1-D tensor, which means a per-tensor
  or per-column quantization. If it's a 1-D tensor, its number of
  elements should be equal to the number of columns of input 'B'.
* **C** (optional, heterogeneous) - **TC**:
  Optional input tensor C. If not specified, the computation is done
  as if C is a scalar 0. The shape of C should be unidirectional
  broadcastable to (M, N). Its type is int32_t and must be quantized
  with zero_point = 0 and scale = alpha / beta * a_scale * b_scale.
* **y_scale** (optional, heterogeneous) - **T**:
  Scale of output 'Y'. It is a scalar, which means a per-tensor
  quantization. It is optional. The output is full precision(float32)
  if it is not provided. Or the output is quantized.
* **y_zero_point** (optional, heterogeneous) - **TYZ**:
  Zero point tensor for output 'Y'. It is a scalar, which means a per-
  tensor quantization. It is optional. The output is full
  precision(float32) if it is not provided. Or the output is
  quantized.

**Outputs**

* **Y** (heterogeneous) - **TY**:
  Output tensor of shape (M, N).

**Examples**
