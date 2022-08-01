
.. _l-onnx-doc-SequenceMap:

===========
SequenceMap
===========

.. contents::
    :local:


.. _l-onnx-op-sequencemap-17:

SequenceMap - 17
================

**Version**

* **name**: `SequenceMap (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceMap>`_
* **domain**: **main**
* **since_version**: **17**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 17**.

**Summary**

Applies a sub-graph to each sample in the input sequence(s).

Inputs can be either tensors or sequences, with the exception of the first input which must
be a sequence. The length of the first input sequence will determine the number of samples in the
outputs. Any other sequence inputs should have the same number of samples. The number of inputs
and outputs, should match the one of the subgraph.

For each i-th element in the output, a sample will be extracted from the input sequence(s) at
the i-th position and the sub-graph will be applied to it.
The outputs will contain the outputs of the sub-graph for each sample, in the same order as in
the input.

This operator assumes that processing each sample is independent and could executed in parallel
or in any order. Users cannot expect any specific ordering in which each subgraph is computed.

**Attributes**

* **body** (required):
  The graph to be run for each sample in the sequence(s). It should
  have as many inputs and outputs as inputs and outputs to the
  SequenceMap function.

**Inputs**

Between 1 and 2147483647 inputs.

* **input_sequence** (heterogeneous) - **S**:
  Input sequence.
* **additional_inputs** (variadic) - **V**:
  Additional inputs to the graph

**Outputs**

Between 1 and 2147483647 outputs.

* **out_sequence** (variadic) - **S**:
  Output sequence(s)

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
  Constrain input types to any sequence type.
* **V** in (
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
  seq(tensor(uint8)),
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
  Constrain to any tensor or sequence type.

**Examples**
