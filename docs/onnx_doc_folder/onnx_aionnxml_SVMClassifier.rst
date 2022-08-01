
.. _l-onnx-docai.onnx.ml-SVMClassifier:

==========================
ai.onnx.ml - SVMClassifier
==========================

.. contents::
    :local:


.. _l-onnx-opai-onnx-ml-svmclassifier-1:

SVMClassifier - 1 (ai.onnx.ml)
==============================

**Version**

* **name**: `SVMClassifier (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.SVMClassifier>`_
* **domain**: **ai.onnx.ml**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1 of domain ai.onnx.ml**.

**Summary**

Support Vector Machine classifier

**Attributes**

* **classlabels_ints**:
  Class labels if using integer labels.<br>One and only one of the
  'classlabels_*' attributes must be defined.
* **classlabels_strings**:
  Class labels if using string labels.<br>One and only one of the
  'classlabels_*' attributes must be defined.
* **coefficients**:

* **kernel_params**:
  List of 3 elements containing gamma, coef0, and degree, in that
  order. Zero if unused for the kernel.
* **kernel_type**:
  The kernel type, one of 'LINEAR,' 'POLY,' 'RBF,' 'SIGMOID'. Default value is ``'LINEAR'``.
* **post_transform**:
  Indicates the transform to apply to the score. <br>One of 'NONE,'
  'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT' Default value is ``'NONE'``.
* **prob_a**:
  First set of probability coefficients.
* **prob_b**:
  Second set of probability coefficients. This array must be same size
  as prob_a.<br>If these are provided then output Z are probability
  estimates, otherwise they are raw scores.
* **rho**:

* **support_vectors**:

* **vectors_per_class**:

**Inputs**

* **X** (heterogeneous) - **T1**:
  Data to be classified.

**Outputs**

* **Y** (heterogeneous) - **T2**:
  Classification outputs (one class per example).
* **Z** (heterogeneous) - **tensor(float)**:
  Class scores (one per class per example), if prob_a and prob_b are
  provided they are probabilities for each class, otherwise they are
  raw scores.

**Type Constraints**

* **T1** in (
  tensor(double),
  tensor(float),
  tensor(int32),
  tensor(int64)
  ):
  The input must be a tensor of a numeric type, either [C] or [N,C].
* **T2** in (
  tensor(int64),
  tensor(string)
  ):
  The output type will be a tensor of strings or integers, depending
  on which of the classlabels_* attributes is used. Its size will
  match the bactch size of the input.

**Examples**
