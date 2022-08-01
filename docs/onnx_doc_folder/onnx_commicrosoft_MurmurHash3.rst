
.. _l-onnx-doccom.microsoft-MurmurHash3:

===========================
com.microsoft - MurmurHash3
===========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-murmurhash3-1:

MurmurHash3 - 1 (com.microsoft)
===============================

**Version**

* **name**: `MurmurHash3 (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.MurmurHash3>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

The underlying implementation is MurmurHash3_x86_32 generating low latency 32bits hash suitable for implementing lookup tables, Bloom filters, count min sketch or feature hashing.

**Attributes**

* **positive**:
  If value is 1, output type is uint32_t, else int32_t. Default value
  is 1. Default value is ``?``.
* **seed**:
  Seed for the hashing algorithm, unsigned 32-bit integer, default to
  0. Default value is ``?``.

**Inputs**

* **X** (heterogeneous) - **T1**:
  An input tensor to hash.

**Outputs**

* **Y** (heterogeneous) - **T2**:
  32-bit hash value.

**Examples**
