
.. _l-onnx-doc-SequenceAt:

==========
SequenceAt
==========

.. contents::
    :local:


.. _l-onnx-op-sequenceat-11:

SequenceAt - 11
===============

**Version**

* **name**: `SequenceAt (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceAt>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

Outputs a tensor copy from the tensor at 'position' in 'input_sequence'.
Accepted range for 'position' is in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.

**Inputs**

* **input_sequence** (heterogeneous) - **S**:
  Input sequence.
* **position** (heterogeneous) - **I**:
  Position of the tensor in the sequence. Negative value means
  counting positions from the back. Accepted range in `[-n, n - 1]`,
  where `n` is the number of tensors in 'input_sequence'. It is an
  error if any of the index values are out of bounds. It must be a
  scalar(tensor of empty shape).

**Outputs**

* **tensor** (heterogeneous) - **T**:
  Output tensor at the specified position in the input sequence.

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
  Constrain to any tensor type.
* **I** in (
  tensor(int32),
  tensor(int64)
  ):
  Constrain position to integral tensor. It must be a scalar(tensor of
  empty shape).

**Examples**
