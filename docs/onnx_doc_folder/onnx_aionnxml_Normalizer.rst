
.. _l-onnx-docai.onnx.ml-Normalizer:

=======================
ai.onnx.ml - Normalizer
=======================

.. contents::
    :local:


.. _l-onnx-opai-onnx-ml-normalizer-1:

Normalizer - 1 (ai.onnx.ml)
===========================

**Version**

* **name**: `Normalizer (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.Normalizer>`_
* **domain**: **ai.onnx.ml**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: False

This version of the operator has been available
**since version 1 of domain ai.onnx.ml**.

**Summary**

    Normalize the input.  There are three normalization modes, which have the corresponding formulas,
    defined using element-wise infix operators '/' and '^' and tensor-wide functions 'max' and 'sum':

    Max: Y = X / max(X)

    L1:  Y = X / sum(X)

    L2:  Y = sqrt(X^2 / sum(X^2)}

    In all modes, if the divisor is zero, Y == X.

    For batches, that is, [N,C] tensors, normalization is done along the C axis. In other words, each row
    of the batch is normalized independently.

**Attributes**

* **norm**:
  One of 'MAX,' 'L1,' 'L2' Default value is ``'MAX'``.

**Inputs**

* **X** (heterogeneous) - **T**:
  Data to be encoded, a tensor of shape [N,C] or [C]

**Outputs**

* **Y** (heterogeneous) - **tensor(float)**:
  Encoded output data

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(int32),
  tensor(int64)
  ):
  The input must be a tensor of a numeric type.

**Examples**
