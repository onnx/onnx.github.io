
.. _l-onnx-docai.onnx.ml-CastMap:

====================
ai.onnx.ml - CastMap
====================

.. contents::
    :local:


.. _l-onnx-opai-onnx-ml-castmap-1:

CastMap - 1 (ai.onnx.ml)
========================

**Version**

* **name**: `CastMap (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.CastMap>`_
* **domain**: **ai.onnx.ml**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1 of domain ai.onnx.ml**.

**Summary**

Converts a map to a tensor.
The map key must be an int64 and the values will be ordered
in ascending order based on this key.
The operator supports dense packing or sparse packing.
If using sparse packing, the key cannot exceed the max_map-1 value.

**Attributes**

* **cast_to**:
  A string indicating the desired element type of the output tensor,
  one of 'TO_FLOAT', 'TO_STRING', 'TO_INT64'. Default value is ``'TO_FLOAT'``.
* **map_form**:
  Indicates whether to only output as many values as are in the input
  (dense), or position the input based on using the key of the map as
  the index of the output (sparse).<br>One of 'DENSE', 'SPARSE'. Default value is ``'DENSE'``.
* **max_map**:
  If the value of map_form is 'SPARSE,' this attribute indicates the
  total length of the output tensor. Default value is ``1``.

**Inputs**

* **X** (heterogeneous) - **T1**:
  The input map that is to be cast to a tensor

**Outputs**

* **Y** (heterogeneous) - **T2**:
  A tensor representing the same data as the input map, ordered by
  their keys

**Type Constraints**

* **T1** in (
  map(int64, float),
  map(int64, string)
  ):
  The input must be an integer map to either string or float.
* **T2** in (
  tensor(float),
  tensor(int64),
  tensor(string)
  ):
  The output is a 1-D tensor of string, float, or integer.

**Examples**
