
.. _l-onnx-doc-SequenceConstruct:

=================
SequenceConstruct
=================

.. contents::
    :local:


.. _l-onnx-op-sequenceconstruct-11:

SequenceConstruct - 11
======================

**Version**

* **name**: `SequenceConstruct (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceConstruct>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

Construct a tensor sequence containing 'inputs' tensors.
All tensors in 'inputs' must have the same data type.

**Inputs**

Between 1 and 2147483647 inputs.

* **inputs** (variadic, heterogeneous) - **T**:
  Tensors.

**Outputs**

* **output_sequence** (heterogeneous) - **S**:
  Sequence enclosing the input tensors.

**Type Constraints**

* **T** in (
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
  Constrain input types to any tensor type.
* **S** in (
  seq(tensor(bool)),
  seq(tensor(complex128)),
  seq(tensor(complex64)),
  seq(tensor(double)),
  seq(tensor(float)),
  seq(tensor(float16)),
  seq(tensor(int16)),
  seq(tensor(int32)),
  seq(tensor(int64)),
  seq(tensor(int8)),
  seq(tensor(string)),
  seq(tensor(uint16)),
  seq(tensor(uint32)),
  seq(tensor(uint64)),
  seq(tensor(uint8))
  ):
  Constrain output types to any tensor type.

**Examples**
