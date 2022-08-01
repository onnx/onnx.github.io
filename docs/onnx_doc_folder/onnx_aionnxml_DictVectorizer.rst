
.. _l-onnx-docai.onnx.ml-DictVectorizer:

===========================
ai.onnx.ml - DictVectorizer
===========================

.. contents::
    :local:


.. _l-onnx-opai-onnx-ml-dictvectorizer-1:

DictVectorizer - 1 (ai.onnx.ml)
===============================

**Version**

* **name**: `DictVectorizer (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.DictVectorizer>`_
* **domain**: **ai.onnx.ml**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1 of domain ai.onnx.ml**.

**Summary**

Uses an index mapping to convert a dictionary to an array.

Given a dictionary, each key is looked up in the vocabulary attribute corresponding to
the key type. The index into the vocabulary array at which the key is found is then
used to index the output 1-D tensor 'Y' and insert into it the value found in the dictionary 'X'.

The key type of the input map must correspond to the element type of the defined vocabulary attribute.
Therefore, the output array will be equal in length to the index mapping vector parameter.
All keys in the input dictionary must be present in the index mapping vector.
For each item in the input dictionary, insert its value in the output array.
Any keys not present in the input dictionary, will be zero in the output array.

For example: if the ``string_vocabulary`` parameter is set to ``["a", "c", "b", "z"]``,
then an input of ``{"a": 4, "c": 8}`` will produce an output of ``[4, 8, 0, 0]``.

**Attributes**

* **int64_vocabulary**:
  An integer vocabulary array.<br>One and only one of the vocabularies
  must be defined.
* **string_vocabulary**:
  A string vocabulary array.<br>One and only one of the vocabularies
  must be defined.

**Inputs**

* **X** (heterogeneous) - **T1**:
  A dictionary.

**Outputs**

* **Y** (heterogeneous) - **T2**:
  A 1-D tensor holding values from the input dictionary.

**Type Constraints**

* **T1** in (
  map(int64, double),
  map(int64, float),
  map(int64, string),
  map(string, double),
  map(string, float),
  map(string, int64)
  ):
  The input must be a map from strings or integers to either strings
  or a numeric type. The key and value types cannot be the same.
* **T2** in (
  tensor(double),
  tensor(float),
  tensor(int64),
  tensor(string)
  ):
  The output will be a tensor of the value type of the input map. It's
  shape will be [1,C], where C is the length of the input dictionary.

**Examples**
