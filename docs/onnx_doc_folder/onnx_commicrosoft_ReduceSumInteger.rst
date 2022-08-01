
.. _l-onnx-doccom.microsoft-ReduceSumInteger:

================================
com.microsoft - ReduceSumInteger
================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-reducesuminteger-1:

ReduceSumInteger - 1 (com.microsoft)
====================================

**Version**

* **name**: `ReduceSumInteger (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.ReduceSumInteger>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Computes the sum of the low-precision input tensor's element along the provided axes.
The resulting tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0,
then the resulting tensor have the reduced dimension pruned. The above behavior is similar to numpy,
with the exception that numpy default keepdims to False instead of True.

**Attributes**

* **axes** (required):
  A list of integers, along which to reduce. The default is to reduce
  over all the dimensions of the input tensor. Default value is ``?``.
* **keepdims** (required):
  Keep the reduced dimension or not, default 1 mean keep reduced
  dimension. Default value is ``?``.

**Inputs**

* **data** (heterogeneous) - **T1**:
  An input tensor.

**Outputs**

* **reduced** (heterogeneous) - **T2**:
  Reduced output tensor.

**Examples**
