
.. _l-onnx-doc-CastLike:

========
CastLike
========

.. contents::
    :local:


.. _l-onnx-op-castlike-15:

CastLike - 15
=============

**Version**

* **name**: `CastLike (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#CastLike>`_
* **domain**: **main**
* **since_version**: **15**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 15**.

**Summary**

The operator casts the elements of a given input tensor (the first input) to
the same data type as the elements of the second input tensor.
See documentation of the Cast operator for further details.

**Inputs**

* **input** (heterogeneous) - **T1**:
  Input tensor to be cast.
* **target_type** (heterogeneous) - **T2**:
  The (first) input tensor will be cast to produce a tensor of the
  same type as this (second input) tensor.

**Outputs**

* **output** (heterogeneous) - **T2**:
  Output tensor produced by casting the first input tensor to have the
  same type as the second input tensor.

**Type Constraints**

* **T1** in (
  tensor(bfloat16),
  tensor(bool),
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
  Constrain input types. Casting from complex is not supported.
* **T2** in (
  tensor(bfloat16),
  tensor(bool),
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
  Constrain output types. Casting to complex is not supported.

**Examples**
