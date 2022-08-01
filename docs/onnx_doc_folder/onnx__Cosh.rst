
.. _l-onnx-doc-Cosh:

====
Cosh
====

.. contents::
    :local:


.. _l-onnx-op-cosh-9:

Cosh - 9
========

**Version**

* **name**: `Cosh (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cosh>`_
* **domain**: **main**
* **since_version**: **9**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 9**.

**Summary**

Calculates the hyperbolic cosine of the given input tensor element-wise.

**Inputs**

* **input** (heterogeneous) - **T**:
  Input tensor

**Outputs**

* **output** (heterogeneous) - **T**:
  The hyperbolic cosine values of the input tensor computed element-
  wise

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.

**Examples**
