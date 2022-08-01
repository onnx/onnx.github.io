
.. _l-onnx-docai.onnx.ml-ZipMap:

===================
ai.onnx.ml - ZipMap
===================

.. contents::
    :local:


.. _l-onnx-opai-onnx-ml-zipmap-1:

ZipMap - 1 (ai.onnx.ml)
=======================

**Version**

* **name**: `ZipMap (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.ZipMap>`_
* **domain**: **ai.onnx.ml**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1 of domain ai.onnx.ml**.

**Summary**

Creates a map from the input and the attributes.

The values are provided by the input tensor, while the keys are specified by the attributes.
Must provide keys in either classlabels_strings or classlabels_int64s (but not both).

The columns of the tensor correspond one-by-one to the keys specified by the attributes. There must be as many columns as keys.

**Attributes**

* **classlabels_int64s**:
  The keys when using int keys.<br>One and only one of the
  'classlabels_*' attributes must be defined.
* **classlabels_strings**:
  The keys when using string keys.<br>One and only one of the
  'classlabels_*' attributes must be defined.

**Inputs**

* **X** (heterogeneous) - **tensor(float)**:
  The input values

**Outputs**

* **Z** (heterogeneous) - **T**:
  The output map

**Type Constraints**

* **T** in (
  seq(map(int64, float)),
  seq(map(string, float))
  ):
  The output will be a sequence of string or integer maps to float.

**Examples**
