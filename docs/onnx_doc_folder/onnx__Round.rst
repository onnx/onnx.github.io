
.. _l-onnx-doc-Round:

=====
Round
=====

.. contents::
    :local:


.. _l-onnx-op-round-11:

Round - 11
==========

**Version**

* **name**: `Round (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Round>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

Round takes one input Tensor and rounds the values, element-wise, meaning
it finds the nearest integer for each value.
In case of halfs, the rule is to round them to the nearest even integer.
The output tensor has the same shape and type as the input.

Examples:
::

    round([0.9]) = [1.0]
    round([2.5]) = [2.0]
    round([2.3]) = [2.0]
    round([1.5]) = [2.0]
    round([-4.5]) = [-4.0]

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
