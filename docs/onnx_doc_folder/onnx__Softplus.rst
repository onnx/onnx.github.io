
.. _l-onnx-doc-Softplus:

========
Softplus
========

.. contents::
    :local:


.. _l-onnx-op-softplus-1:

Softplus - 1
============

**Version**

* **name**: `Softplus (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softplus>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1**.

**Summary**

Softplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to
the tensor elementwise.

**Inputs**

* **X** (heterogeneous) - **T**:
  1D input tensor

**Outputs**

* **Y** (heterogeneous) - **T**:
  1D input tensor

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.

**Examples**
