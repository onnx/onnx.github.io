
.. _l-onnx-docai.onnx.ml-SVMRegressor:

=========================
ai.onnx.ml - SVMRegressor
=========================

.. contents::
    :local:


.. _l-onnx-opai-onnx-ml-svmregressor-1:

SVMRegressor - 1 (ai.onnx.ml)
=============================

**Version**

* **name**: `SVMRegressor (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.SVMRegressor>`_
* **domain**: **ai.onnx.ml**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: False

This version of the operator has been available
**since version 1 of domain ai.onnx.ml**.

**Summary**

Support Vector Machine regression prediction and one-class SVM anomaly detection.

**Attributes**

* **coefficients**:
  Support vector coefficients.
* **kernel_params**:
  List of 3 elements containing gamma, coef0, and degree, in that
  order. Zero if unused for the kernel.
* **kernel_type**:
  The kernel type, one of 'LINEAR,' 'POLY,' 'RBF,' 'SIGMOID'. Default value is ``'LINEAR'``.
* **n_supports**:
  The number of support vectors. Default value is ``0``.
* **one_class**:
  Flag indicating whether the regression is a one-class SVM or not. Default value is ``0``.
* **post_transform**:
  Indicates the transform to apply to the score. <br>One of 'NONE,'
  'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT.' Default value is ``'NONE'``.
* **rho**:

* **support_vectors**:
  Chosen support vectors

**Inputs**

* **X** (heterogeneous) - **T**:
  Data to be regressed.

**Outputs**

* **Y** (heterogeneous) - **tensor(float)**:
  Regression outputs (one score per target per example).

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(int32),
  tensor(int64)
  ):
  The input type must be a tensor of a numeric type, either [C] or
  [N,C].

**Examples**
