
.. _l-onnx-docai.onnx.ml-ArrayFeatureExtractor:

==================================
ai.onnx.ml - ArrayFeatureExtractor
==================================

.. contents::
    :local:


.. _l-onnx-opai-onnx-ml-arrayfeatureextractor-1:

ArrayFeatureExtractor - 1 (ai.onnx.ml)
======================================

**Version**

* **name**: `ArrayFeatureExtractor (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.ArrayFeatureExtractor>`_
* **domain**: **ai.onnx.ml**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: False

This version of the operator has been available
**since version 1 of domain ai.onnx.ml**.

**Summary**

Select elements of the input tensor based on the indices passed.

The indices are applied to the last axes of the tensor.

**Inputs**

* **X** (heterogeneous) - **T**:
  Data to be selected
* **Y** (heterogeneous) - **tensor(int64)**:
  The indices, based on 0 as the first index of any dimension.

**Outputs**

* **Z** (heterogeneous) - **T**:
  Selected output data as an array

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(int32),
  tensor(int64),
  tensor(string)
  ):
  The input must be a tensor of a numeric type or string. The output
  will be of the same tensor type.

**Examples**
