
.. _l-onnx-doc-SplitToSequence:

===============
SplitToSequence
===============

.. contents::
    :local:


.. _l-onnx-op-splittosequence-11:

SplitToSequence - 11
====================

**Version**

* **name**: `SplitToSequence (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#SplitToSequence>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

Split a tensor into a sequence of tensors, along the specified
'axis'. Lengths of the parts can be specified using argument 'split'.
'split' must contain only positive numbers.
'split' is either a scalar (tensor of empty shape), or a 1-D tensor.
If 'split' is a scalar, then 'input' will be split into equally sized chunks(if possible).
Last chunk will be smaller if the 'input' size along the given axis 'axis' is not divisible
by 'split'.
Otherwise, the tensor is split into 'size(split)' chunks, with lengths of the parts on 'axis'
specified in 'split'. In this scenario, the sum of entries in 'split' must be equal to the
dimension size of input tensor on 'axis'.

**Attributes**

* **axis**:
  Which axis to split on. A negative value means counting dimensions
  from the back. Accepted range is [-rank, rank-1]. Default value is ``0``.
* **keepdims**:
  Keep the split dimension or not. Default 1, which means we keep
  split dimension. If input 'split' is specified, this attribute is
  ignored. Default value is ``1``.

**Inputs**

Between 1 and 2 inputs.

* **input** (heterogeneous) - **T**:
  The tensor to split
* **split** (optional, heterogeneous) - **I**:
  Length of each output. It can be either a scalar(tensor of empty
  shape), or a 1-D tensor. All values must be >= 0.

**Outputs**

* **output_sequence** (heterogeneous) - **S**:
  One or more outputs forming a sequence of tensors after splitting

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
  Constrain input types to all tensor types.
* **I** in (
  tensor(int32),
  tensor(int64)
  ):
  Constrain split size to integral tensor.
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
  Constrain output types to all tensor types.

**Examples**
