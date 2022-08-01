
.. _l-onnx-doccom.microsoft-FusedGemm:

=========================
com.microsoft - FusedGemm
=========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-fusedgemm-1:

FusedGemm - 1 (com.microsoft)
=============================

**Version**

* **name**: `FusedGemm (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.FusedGemm>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

The FusedGemm operator schema is the same as Gemm besides it includes attributes
activation and leaky_relu_alpha.

**Attributes**

* **activation**:
 Default value is ``?``.
* **activation_alpha**:
 Default value is ``?``.
* **activation_beta**:
 Default value is ``?``.
* **activation_gamma**:
 Default value is ``?``.
* **alpha**:
  Scalar multiplier for the product of input tensors A * B. Default value is ``?``.
* **beta**:
  Scalar multiplier for input tensor C. Default value is ``?``.
* **transA**:
  Whether A should be transposed Default value is ``?``.
* **transB**:
  Whether B should be transposed Default value is ``?``.

**Inputs**

* **A** (heterogeneous) - **T**:
  Input tensor A. The shape of A should be (M, K) if transA is 0, or
  (K, M) if transA is non-zero.
* **B** (heterogeneous) - **T**:
  Input tensor B. The shape of B should be (K, N) if transB is 0, or
  (N, K) if transB is non-zero.
* **C** (heterogeneous) - **T**:
  Input tensor C. The shape of C should be unidirectional
  broadcastable to (M, N).

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output tensor of shape (M, N).

**Examples**
