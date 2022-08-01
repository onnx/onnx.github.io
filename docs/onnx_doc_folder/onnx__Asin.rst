
.. _l-onnx-doc-Asin:

====
Asin
====

.. contents::
    :local:


.. _l-onnx-op-asin-7:

Asin - 7
========

**Version**

* **name**: `Asin (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Asin>`_
* **domain**: **main**
* **since_version**: **7**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 7**.

**Summary**

Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.

**Inputs**

* **input** (heterogeneous) - **T**:
  Input tensor

**Outputs**

* **output** (heterogeneous) - **T**:
  The arcsine of the input tensor computed element-wise

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.

**Examples**
