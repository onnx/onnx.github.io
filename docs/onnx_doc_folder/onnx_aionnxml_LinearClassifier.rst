
.. _l-onnx-docai.onnx.ml-LinearClassifier:

=============================
ai.onnx.ml - LinearClassifier
=============================

.. contents::
    :local:


.. _l-onnx-opai-onnx-ml-linearclassifier-1:

LinearClassifier - 1 (ai.onnx.ml)
=================================

**Version**

* **name**: `LinearClassifier (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.LinearClassifier>`_
* **domain**: **ai.onnx.ml**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1 of domain ai.onnx.ml**.

**Summary**

Linear classifier

**Attributes**

* **classlabels_ints**:
  Class labels when using integer labels. One and only one
  'classlabels' attribute must be defined.
* **classlabels_strings**:
  Class labels when using string labels. One and only one
  'classlabels' attribute must be defined.
* **coefficients** (required):
  A collection of weights of the model(s).
* **intercepts**:
  A collection of intercepts.
* **multi_class**:
  Indicates whether to do OvR or multinomial (0=OvR is the default). Default value is ``0``.
* **post_transform**:
  Indicates the transform to apply to the scores vector.<br>One of
  'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT' Default value is ``'NONE'``.

**Inputs**

* **X** (heterogeneous) - **T1**:
  Data to be classified.

**Outputs**

* **Y** (heterogeneous) - **T2**:
  Classification outputs (one class per example).
* **Z** (heterogeneous) - **tensor(float)**:
  Classification scores ([N,E] - one score for each class and example

**Type Constraints**

* **T1** in (
  tensor(double),
  tensor(float),
  tensor(int32),
  tensor(int64)
  ):
  The input must be a tensor of a numeric type, and of shape [N,C] or
  [C]. In the latter case, it will be treated as [1,C]
* **T2** in (
  tensor(int64),
  tensor(string)
  ):
  The output will be a tensor of strings or integers.

**Examples**
