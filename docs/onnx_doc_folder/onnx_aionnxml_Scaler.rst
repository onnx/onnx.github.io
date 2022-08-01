
.. _l-onnx-docai.onnx.ml-Scaler:

===================
ai.onnx.ml - Scaler
===================

.. contents::
    :local:


.. _l-onnx-opai-onnx-ml-scaler-1:

Scaler - 1 (ai.onnx.ml)
=======================

**Version**

* **name**: `Scaler (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.Scaler>`_
* **domain**: **ai.onnx.ml**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: False

This version of the operator has been available
**since version 1 of domain ai.onnx.ml**.

**Summary**

Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.

**Attributes**

* **offset**:
  First, offset by this.<br>Can be length of features in an [N,F]
  tensor or length 1, in which case it applies to all features,
  regardless of dimension count.
* **scale**:
  Second, multiply by this.<br>Can be length of features in an [N,F]
  tensor or length 1, in which case it applies to all features,
  regardless of dimension count.<br>Must be same length as 'offset'

**Inputs**

* **X** (heterogeneous) - **T**:
  Data to be scaled.

**Outputs**

* **Y** (heterogeneous) - **tensor(float)**:
  Scaled output data.

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(int32),
  tensor(int64)
  ):
  The input must be a tensor of a numeric type.

**Examples**
