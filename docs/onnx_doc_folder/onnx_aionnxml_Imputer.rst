
.. _l-onnx-docai.onnx.ml-Imputer:

====================
ai.onnx.ml - Imputer
====================

.. contents::
    :local:


.. _l-onnx-opai-onnx-ml-imputer-1:

Imputer - 1 (ai.onnx.ml)
========================

**Version**

* **name**: `Imputer (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.Imputer>`_
* **domain**: **ai.onnx.ml**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: False

This version of the operator has been available
**since version 1 of domain ai.onnx.ml**.

**Summary**

Replaces inputs that equal one value with another, leaving all other elements alone.

This operator is typically used to replace missing values in situations where they have a canonical
representation, such as -1, 0, NaN, or some extreme value.

One and only one of imputed_value_floats or imputed_value_int64s should be defined -- floats if the input tensor
holds floats, integers if the input tensor holds integers. The imputed values must all fit within the
width of the tensor element type. One and only one of the replaced_value_float or replaced_value_int64 should be defined,
which one depends on whether floats or integers are being processed.

The imputed_value attribute length can be 1 element, or it can have one element per input feature.
In other words, if the input tensor has the shape [*,F], then the length of the attribute array may be 1 or F. If it is 1, then it is broadcast along the last dimension and applied to each feature.

**Attributes**

* **imputed_value_floats**:
  Value(s) to change to
* **imputed_value_int64s**:
  Value(s) to change to.
* **replaced_value_float**:
  A value that needs replacing. Default value is ``0.0``.
* **replaced_value_int64**:
  A value that needs replacing. Default value is ``0``.

**Inputs**

* **X** (heterogeneous) - **T**:
  Data to be processed.

**Outputs**

* **Y** (heterogeneous) - **T**:
  Imputed output data

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(int32),
  tensor(int64)
  ):
  The input type must be a tensor of a numeric type, either [N,C] or
  [C]. The output type will be of the same tensor type and shape.

**Examples**
