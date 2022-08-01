
.. _l-onnx-doc-HammingWindow:

=============
HammingWindow
=============

.. contents::
    :local:


.. _l-onnx-op-hammingwindow-17:

HammingWindow - 17
==================

**Version**

* **name**: `HammingWindow (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#HammingWindow>`_
* **domain**: **main**
* **since_version**: **17**
* **function**: True
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 17**.

**Summary**

Generates a Hamming window as described in the paper https://ieeexplore.ieee.org/document/1455106.

**Attributes**

* **output_datatype**:
  The data type of the output tensor. Strictly must be one of the
  values from DataType enum in TensorProto whose values correspond to
  T2. The default value is 1 = FLOAT. Default value is ``1``.
* **periodic**:
  If 1, returns a window to be used as periodic function. If 0, return
  a symmetric window. When 'periodic' is specified, hann computes a
  window of length size + 1 and returns the first size points. The
  default value is 1. Default value is ``1``.

**Inputs**

* **size** (heterogeneous) - **T1**:
  A scalar value indicating the length of the window.

**Outputs**

* **output** (heterogeneous) - **T2**:
  A Hamming window with length: size. The output has the shape:
  [size].

**Type Constraints**

* **T1** in (
  tensor(int32),
  tensor(int64)
  ):
  Constrain the input size to int64_t.
* **T2** in (
  tensor(bfloat16),
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int16),
  tensor(int32),
  tensor(int64),
  tensor(int8),
  tensor(uint16),
  tensor(uint32),
  tensor(uint64),
  tensor(uint8)
  ):
  Constrain output types to numeric tensors.

**Examples**
