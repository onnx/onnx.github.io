
.. _l-onnx-doc-SequenceLength:

==============
SequenceLength
==============

.. contents::
    :local:


.. _l-onnx-op-sequencelength-11:

SequenceLength - 11
===================

**Version**

* **name**: `SequenceLength (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceLength>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

Produces a scalar(tensor of empty shape) containing the number of tensors in 'input_sequence'.

**Inputs**

* **input_sequence** (heterogeneous) - **S**:
  Input sequence.

**Outputs**

* **length** (heterogeneous) - **I**:
  Length of input sequence. It must be a scalar(tensor of empty
  shape).

**Type Constraints**

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
  Constrain to any tensor type.
* **I** in (
  tensor(int64)
  ):
  Constrain output to integral tensor. It must be a scalar(tensor of
  empty shape).

**Examples**
