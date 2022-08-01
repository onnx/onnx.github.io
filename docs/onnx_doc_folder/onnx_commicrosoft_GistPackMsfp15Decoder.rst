
.. _l-onnx-doccom.microsoft-GistPackMsfp15Decoder:

=====================================
com.microsoft - GistPackMsfp15Decoder
=====================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-gistpackmsfp15decoder-1:

GistPackMsfp15Decoder - 1 (com.microsoft)
=========================================

**Version**

* **name**: `GistPackMsfp15Decoder (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.GistPackMsfp15Decoder>`_
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
  compresssed input

**Outputs**

* **Y** (heterogeneous) - **T**:
  uncompressed output

**Examples**
