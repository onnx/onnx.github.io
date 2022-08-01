
.. _l-onnx-doc-Tan:

===
Tan
===

.. contents::
    :local:


.. _l-onnx-op-tan-7:

Tan - 7
=======

**Version**

* **name**: `Tan (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tan>`_
* **domain**: **main**
* **since_version**: **7**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 7**.

**Summary**

Calculates the tangent of the given input tensor, element-wise.

**Inputs**

* **input** (heterogeneous) - **T**:
  Input tensor

**Outputs**

* **output** (heterogeneous) - **T**:
  The tangent of the input tensor computed element-wise

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.

**Examples**
