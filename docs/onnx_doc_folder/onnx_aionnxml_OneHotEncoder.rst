
.. _l-onnx-docai.onnx.ml-OneHotEncoder:

==========================
ai.onnx.ml - OneHotEncoder
==========================

.. contents::
    :local:


.. _l-onnx-opai-onnx-ml-onehotencoder-1:

OneHotEncoder - 1 (ai.onnx.ml)
==============================

**Version**

* **name**: `OneHotEncoder (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.OneHotEncoder>`_
* **domain**: **ai.onnx.ml**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: False

This version of the operator has been available
**since version 1 of domain ai.onnx.ml**.

**Summary**

Replace each input element with an array of ones and zeros, where a single
one is placed at the index of the category that was passed in. The total category count
will determine the size of the extra dimension of the output array Y.

For example, if we pass a tensor with a single value of 4, and a category count of 8,
the output will be a tensor with ``[0,0,0,0,1,0,0,0]``.

This operator assumes every input feature is from the same set of categories.

If the input is a tensor of float, int32, or double, the data will be cast
to integers and the cats_int64s category list will be used for the lookups.

**Attributes**

* **cats_int64s**:
  List of categories, ints.<br>One and only one of the 'cats_*'
  attributes must be defined.
* **cats_strings**:
  List of categories, strings.<br>One and only one of the 'cats_*'
  attributes must be defined.
* **zeros**:
  If true and category is not present, will return all zeros; if false
  and a category if not found, the operator will fail. Default value is ``1``.

**Inputs**

* **X** (heterogeneous) - **T**:
  Data to be encoded.

**Outputs**

* **Y** (heterogeneous) - **tensor(float)**:
  Encoded output data, having one more dimension than X.

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(int32),
  tensor(int64),
  tensor(string)
  ):
  The input must be a tensor of a numeric type.

**Examples**
