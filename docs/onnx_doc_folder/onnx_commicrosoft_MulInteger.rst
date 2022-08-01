
.. _l-onnx-doccom.microsoft-MulInteger:

==========================
com.microsoft - MulInteger
==========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-mulinteger-1:

MulInteger - 1 (com.microsoft)
==============================

**Version**

* **name**: `MulInteger (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.MulInteger>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Performs element-wise binary quantized multiplication (with Numpy-style broadcasting support).
"This operator supports **multidirectional (i.e., Numpy-style) broadcasting**"
The output of this op is the int32 accumulated result of the mul operation

::

    C (int32) = (A - A_zero_point) * (B - B_zero_point)

**Inputs**

Between 3 and 4 inputs.

* **A** (heterogeneous) - **T**:
  First operand.
* **A_zero_point** (optional, heterogeneous) - **T**:
  Input A zero point. Default value is 0 if it's not specified. It's a
  scalar, which means a per-tensor/layer quantization.
* **B** (heterogeneous) - **T**:
  Second operand.
* **B_zero_point** (optional, heterogeneous) - **T**:
  Input B zero point. Default value is 0 if it's not specified. It's a
  scalar, which means a per-tensor/layer quantization.

**Outputs**

* **C** (heterogeneous) - **T1**:
  Constrain output to 32 bit tensor

**Examples**
