
.. _l-onnx-doccom.microsoft-GistPack16Decoder:

=================================
com.microsoft - GistPack16Decoder
=================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-gistpack16decoder-1:

GistPack16Decoder - 1 (com.microsoft)
=====================================

**Version**

* **name**: `GistPack16Decoder (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.GistPack16Decoder>`_
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
  compressed input

**Outputs**

* **Y** (heterogeneous) - **T**:
  uncompressed output

**Examples**
