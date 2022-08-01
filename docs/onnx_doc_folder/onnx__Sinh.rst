
.. _l-onnx-doc-Sinh:

====
Sinh
====

.. contents::
    :local:


.. _l-onnx-op-sinh-9:

Sinh - 9
========

**Version**

* **name**: `Sinh (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sinh>`_
* **domain**: **main**
* **since_version**: **9**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 9**.

**Summary**

Calculates the hyperbolic sine of the given input tensor element-wise.

**Inputs**

* **input** (heterogeneous) - **T**:
  Input tensor

**Outputs**

* **output** (heterogeneous) - **T**:
  The hyperbolic sine values of the input tensor computed element-wise

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.

**Examples**
