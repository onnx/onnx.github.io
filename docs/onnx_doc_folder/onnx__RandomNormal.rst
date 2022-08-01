
.. _l-onnx-doc-RandomNormal:

============
RandomNormal
============

.. contents::
    :local:


.. _l-onnx-op-randomnormal-1:

RandomNormal - 1
================

**Version**

* **name**: `RandomNormal (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomNormal>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1**.

**Summary**

Generate a tensor with random values drawn from a normal distribution. The shape
of the tensor is specified by the `shape` argument and the parameter of the normal distribution
specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.

**Attributes**

* **dtype**:
  The data type for the elements of the output tensor. Default is
  TensorProto::FLOAT. Default value is ``1``.
* **mean**:
  The mean of the normal distribution. Default value is ``0.0``.
* **scale**:
  The standard deviation of the normal distribution. Default value is ``1.0``.
* **seed**:
  (Optional) Seed to the random generator, if not specified we will
  auto generate one.
* **shape** (required):
  The shape of the output tensor.

**Outputs**

* **output** (heterogeneous) - **T**:
  Output tensor of random values drawn from normal distribution

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain output types to float tensors.

**Examples**
