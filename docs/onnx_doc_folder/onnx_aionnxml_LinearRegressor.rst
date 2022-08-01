
.. _l-onnx-docai.onnx.ml-LinearRegressor:

============================
ai.onnx.ml - LinearRegressor
============================

.. contents::
    :local:


.. _l-onnx-opai-onnx-ml-linearregressor-1:

LinearRegressor - 1 (ai.onnx.ml)
================================

**Version**

* **name**: `LinearRegressor (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.LinearRegressor>`_
* **domain**: **ai.onnx.ml**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: False

This version of the operator has been available
**since version 1 of domain ai.onnx.ml**.

**Summary**

Generalized linear regression evaluation.

If targets is set to 1 (default) then univariate regression is performed.

If targets is set to M then M sets of coefficients must be passed in as a sequence
and M results will be output for each input n in N.

The coefficients array is of length n, and the coefficients for each target are contiguous.
Intercepts are optional but if provided must match the number of targets.

**Attributes**

* **coefficients**:
  Weights of the model(s).
* **intercepts**:
  Weights of the intercepts, if used.
* **post_transform**:
  Indicates the transform to apply to the regression output
  vector.<br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or
  'PROBIT' Default value is ``'NONE'``.
* **targets**:
  The total number of regression targets, 1 if not defined. Default value is ``1``.

**Inputs**

* **X** (heterogeneous) - **T**:
  Data to be regressed.

**Outputs**

* **Y** (heterogeneous) - **tensor(float)**:
  Regression outputs (one per target, per example).

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(int32),
  tensor(int64)
  ):
  The input must be a tensor of a numeric type.

**Examples**
