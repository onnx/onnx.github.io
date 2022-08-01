
.. _l-onnx-doc-RandomUniformLike:

=================
RandomUniformLike
=================

.. contents::
    :local:


.. _l-onnx-op-randomuniformlike-1:

RandomUniformLike - 1
=====================

**Version**

* **name**: `RandomUniformLike (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomUniformLike>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1**.

**Summary**

Generate a tensor with random values drawn from a uniform distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the uniform distribution are specified by `low` and `high`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.

**Attributes**

* **dtype**:
  (Optional) The data type for the elements of the output tensor, if
  not specified, we will use the data type of the input tensor.
* **high**:
  Upper boundary of the output values. Default value is ``1.0``.
* **low**:
  Lower boundary of the output values. Default value is ``0.0``.
* **seed**:
  (Optional) Seed to the random generator, if not specified we will
  auto generate one.

**Inputs**

* **input** (heterogeneous) - **T1**:
  Input tensor to copy shape and optionally type information from.

**Outputs**

* **output** (heterogeneous) - **T2**:
  Output tensor of random values drawn from uniform distribution

**Type Constraints**

* **T1** in (
  tensor(bool),
  tensor(complex128),
  tensor(complex64),
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int16),
  tensor(int32),
  tensor(int64),
  tensor(int8),
  tensor(string),
  tensor(uint16),
  tensor(uint32),
  tensor(uint64),
  tensor(uint8)
  ):
  Constrain to any tensor type. If the dtype attribute is not provided
  this must be a valid output type.
* **T2** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain output types to float tensors.

**Examples**
