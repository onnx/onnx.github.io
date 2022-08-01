
.. _l-onnx-doc-HardSwish:

=========
HardSwish
=========

.. contents::
    :local:


.. _l-onnx-op-hardswish-14:

HardSwish - 14
==============

**Version**

* **name**: `HardSwish (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#HardSwish>`_
* **domain**: **main**
* **since_version**: **14**
* **function**: True
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 14**.

**Summary**

HardSwish takes one input data (Tensor<T>) and produces one output data (Tensor<T>) where
the HardSwish function, y = x * max(0, min(1, alpha * x + beta)) = x * HardSigmoid<alpha, beta>(x),
where alpha = 1/6 and beta = 0.5, is applied to the tensor elementwise.

**Inputs**

* **X** (heterogeneous) - **T**:
  Input tensor

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output tensor

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.

**Examples**
