
.. _l-onnx-docai.onnx.ml-Binarizer:

======================
ai.onnx.ml - Binarizer
======================

.. contents::
    :local:


.. _l-onnx-opai-onnx-ml-binarizer-1:

Binarizer - 1 (ai.onnx.ml)
==========================

**Version**

* **name**: `Binarizer (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.Binarizer>`_
* **domain**: **ai.onnx.ml**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: False

This version of the operator has been available
**since version 1 of domain ai.onnx.ml**.

**Summary**

Maps the values of the input tensor to either 0 or 1, element-wise, based on the outcome of a comparison against a threshold value.

**Attributes**

* **threshold**:
  Values greater than this are mapped to 1, others to 0. Default value is ``0.0``.

**Inputs**

* **X** (heterogeneous) - **T**:
  Data to be binarized

**Outputs**

* **Y** (heterogeneous) - **T**:
  Binarized output data

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(int32),
  tensor(int64)
  ):
  The input must be a tensor of a numeric type. The output will be of
  the same tensor type.

**Examples**
