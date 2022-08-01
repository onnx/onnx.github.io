
.. _l-onnx-doc-Softsign:

========
Softsign
========

.. contents::
    :local:


.. _l-onnx-op-softsign-1:

Softsign - 1
============

**Version**

* **name**: `Softsign (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softsign>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1**.

**Summary**

Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.

**Inputs**

* **input** (heterogeneous) - **T**:
  Input tensor

**Outputs**

* **output** (heterogeneous) - **T**:
  The softsign (x/(1+|x|)) values of the input tensor computed
  element-wise

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.

**Examples**
