
.. _l-onnx-doc-SequenceEmpty:

=============
SequenceEmpty
=============

.. contents::
    :local:


.. _l-onnx-op-sequenceempty-11:

SequenceEmpty - 11
==================

**Version**

* **name**: `SequenceEmpty (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceEmpty>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

Construct an empty tensor sequence, with given data type.

**Attributes**

* **dtype**:
  (Optional) The data type of the tensors in the output sequence. The
  default type is 'float'.

**Outputs**

* **output** (heterogeneous) - **S**:
  Empty sequence.

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
  Constrain output types to any tensor type.

**Examples**
