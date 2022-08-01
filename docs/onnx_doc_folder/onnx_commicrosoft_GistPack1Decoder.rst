
.. _l-onnx-doccom.microsoft-GistPack1Decoder:

================================
com.microsoft - GistPack1Decoder
================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-gistpack1decoder-1:

GistPack1Decoder - 1 (com.microsoft)
====================================

**Version**

* **name**: `GistPack1Decoder (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.GistPack1Decoder>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

**Attributes**

* **to** (required):
  The data type to which the elements of the input tensor are cast.
  Strictly must be one of the types from DataType enum in TensorProto Default value is ``?``.

**Inputs**

* **X** (heterogeneous) - **T1**:
  1 bit compresssed input

**Outputs**

* **Y** (heterogeneous) - **T**:
  uncompressed output

**Examples**
