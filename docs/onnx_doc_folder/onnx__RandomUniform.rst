
.. _l-onnx-doc-RandomUniform:

=============
RandomUniform
=============

.. contents::
    :local:


.. _l-onnx-op-randomuniform-1:

RandomUniform - 1
=================

**Version**

* **name**: `RandomUniform (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomUniform>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1**.

**Summary**

Generate a tensor with random values drawn from a uniform distribution. The shape
of the tensor is specified by the `shape` argument and the range by `low` and `high`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.

**Attributes**

* **dtype**:
  The data type for the elements of the output tensor. If not
  specified, default is TensorProto::FLOAT. Default value is ``1``.
* **high**:
  Upper boundary of the output values. Default value is ``1.0``.
* **low**:
  Lower boundary of the output values. Default value is ``0.0``.
* **seed**:
  (Optional) Seed to the random generator, if not specified we will
  auto generate one.
* **shape** (required):
  The shape of the output tensor.

**Outputs**

* **output** (heterogeneous) - **T**:
  Output tensor of random values drawn from uniform distribution

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain output types to float tensors.

**Examples**
