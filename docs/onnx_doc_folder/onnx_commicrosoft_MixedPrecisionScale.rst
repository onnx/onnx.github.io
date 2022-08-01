
.. _l-onnx-doccom.microsoft-MixedPrecisionScale:

===================================
com.microsoft - MixedPrecisionScale
===================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-mixedprecisionscale-1:

MixedPrecisionScale - 1 (com.microsoft)
=======================================

**Version**

* **name**: `MixedPrecisionScale (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.MixedPrecisionScale>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

MixedPrecisionScale

**Attributes**

* **fuse_outputs**:
  If true, fuse all outputs into one continous buffer. Default value is ``?``.
* **to** (required):
  The data type to which the elements of the input tensor are cast.
  Strictly must be one of the types from DataType enum in TensorProto Default value is ``?``.

**Inputs**

Between 2 and 2147483647 inputs.

* **S** (heterogeneous) - **ScaleT**:
  scale
* **X** (variadic, heterogeneous) - **SrcT**:
  inputs

**Outputs**

Between 1 and 2147483647 outputs.

* **Y** (variadic, heterogeneous) - **DstT**:
  output

**Examples**
