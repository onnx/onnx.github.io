
.. _l-onnx-docai.onnx.ml-FeatureVectorizer:

==============================
ai.onnx.ml - FeatureVectorizer
==============================

.. contents::
    :local:


.. _l-onnx-opai-onnx-ml-featurevectorizer-1:

FeatureVectorizer - 1 (ai.onnx.ml)
==================================

**Version**

* **name**: `FeatureVectorizer (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.FeatureVectorizer>`_
* **domain**: **ai.onnx.ml**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: False

This version of the operator has been available
**since version 1 of domain ai.onnx.ml**.

**Summary**

Concatenates input tensors into one continuous output.

All input shapes are 2-D and are concatenated along the second dimention. 1-D tensors are treated as [1,C].
Inputs are copied to the output maintaining the order of the input arguments.

All inputs must be integers or floats, while the output will be all floating point values.

**Attributes**

* **inputdimensions**:
  The size of each input in the input list

**Inputs**

Between 1 and 2147483647 inputs.

* **X** (variadic, heterogeneous) - **T1**:
  An ordered collection of tensors, all with the same element type.

**Outputs**

* **Y** (heterogeneous) - **tensor(float)**:
  The output array, elements ordered as the inputs.

**Type Constraints**

* **T1** in (
  tensor(double),
  tensor(float),
  tensor(int32),
  tensor(int64)
  ):
  The input type must be a tensor of a numeric type.

**Examples**
