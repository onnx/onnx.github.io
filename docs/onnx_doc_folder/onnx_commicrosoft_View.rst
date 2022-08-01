
.. _l-onnx-doccom.microsoft-View:

====================
com.microsoft - View
====================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-view-1:

View - 1 (com.microsoft)
========================

**Version**

* **name**: `View (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.View>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

View. The output tensors are views of the input, according to the shapes provided.

**Inputs**

Between 2 and 2147483647 inputs.

* **input** (heterogeneous) - **T**:
  Input tensor.
* **shapes** (variadic, heterogeneous) - **tensor(int64)**:
  Shapes of each view output. The shapes must adds up to the input
  buffer size.

**Outputs**

Between 1 and 2147483647 outputs.

* **outputs** (variadic, heterogeneous) - **T**:
  Output tensors viewed according the shapes input. It has a one to
  one mapping to the shapes input

**Examples**
