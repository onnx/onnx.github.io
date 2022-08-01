
.. _l-onnx-doc-Celu:

====
Celu
====

.. contents::
    :local:


.. _l-onnx-op-celu-12:

Celu - 12
=========

**Version**

* **name**: `Celu (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Celu>`_
* **domain**: **main**
* **since_version**: **12**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 12**.

**Summary**

Continuously Differentiable Exponential Linear Units:
Perform the linear unit element-wise on the input tensor X
using formula:

::

    max(0,x) + min(0,alpha*(exp(x/alpha)-1))

**Attributes**

* **alpha**:
  The Alpha value in Celu formula which control the shape of the unit.
  The default value is 1.0. Default value is ``1.0``.

**Inputs**

* **X** (heterogeneous) - **T**:
  Input tensor

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output tensor

**Type Constraints**

* **T** in (
  tensor(float)
  ):
  Constrain input and output types to float32 tensors.

**Examples**
