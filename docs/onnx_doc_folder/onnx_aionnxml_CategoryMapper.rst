
.. _l-onnx-docai.onnx.ml-CategoryMapper:

===========================
ai.onnx.ml - CategoryMapper
===========================

.. contents::
    :local:


.. _l-onnx-opai-onnx-ml-categorymapper-1:

CategoryMapper - 1 (ai.onnx.ml)
===============================

**Version**

* **name**: `CategoryMapper (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.CategoryMapper>`_
* **domain**: **ai.onnx.ml**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1 of domain ai.onnx.ml**.

**Summary**

Converts strings to integers and vice versa.

Two sequences of equal length are used to map between integers and strings,
with strings and integers at the same index detailing the mapping.

Each operator converts either integers to strings or strings to integers, depending
on which default value attribute is provided. Only one default value attribute
should be defined.

If the string default value is set, it will convert integers to strings.
If the int default value is set, it will convert strings to integers.

**Attributes**

* **cats_int64s**:
  The integers of the map. This sequence must be the same length as
  the 'cats_strings' sequence.
* **cats_strings**:
  The strings of the map. This sequence must be the same length as the
  'cats_int64s' sequence
* **default_int64**:
  An integer to use when an input string value is not found in the
  map.<br>One and only one of the 'default_*' attributes must be
  defined. Default value is ``-1``.
* **default_string**:
  A string to use when an input integer value is not found in the
  map.<br>One and only one of the 'default_*' attributes must be
  defined. Default value is ``'_Unused'``.

**Inputs**

* **X** (heterogeneous) - **T1**:
  Input data

**Outputs**

* **Y** (heterogeneous) - **T2**:
  Output data. If strings are input, the output values are integers,
  and vice versa.

**Type Constraints**

* **T1** in (
  tensor(int64),
  tensor(string)
  ):
  The input must be a tensor of strings or integers, either [N,C] or
  [C].
* **T2** in (
  tensor(int64),
  tensor(string)
  ):
  The output is a tensor of strings or integers. Its shape will be the
  same as the input shape.

**Examples**
