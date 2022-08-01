
.. _l-onnx-doc-LpNormalization:

===============
LpNormalization
===============

.. contents::
    :local:


.. _l-onnx-op-lpnormalization-1:

LpNormalization - 1
===================

**Version**

* **name**: `LpNormalization (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#LpNormalization>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1**.

**Summary**

Given a matrix, apply Lp-normalization along the provided axis.

**Attributes**

* **axis**:
  The axis on which to apply normalization, -1 mean last axis. Default value is ``-1``.
* **p**:
  The order of the normalization, only 1 or 2 are supported. Default value is ``2``.

**Inputs**

* **input** (heterogeneous) - **T**:
  Input matrix

**Outputs**

* **output** (heterogeneous) - **T**:
  Matrix after normalization

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.

**Examples**
