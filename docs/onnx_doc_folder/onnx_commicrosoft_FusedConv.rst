
.. _l-onnx-doccom.microsoft-FusedConv:

=========================
com.microsoft - FusedConv
=========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-fusedconv-1:

FusedConv - 1 (com.microsoft)
=============================

**Version**

* **name**: `FusedConv (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.FusedConv>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

The fused convolution operator schema is the same as Conv besides it includes an attribute
activation.

**Attributes**

* **activation**:
 Default value is ``?``.
* **activation_params**:
 Default value is ``?``.
* **auto_pad**:
 Default value is ``?``.
* **dilations**:
 Default value is ``?``.
* **group**:
 Default value is ``?``.
* **kernel_shape**:
 Default value is ``?``.
* **pads**:
 Default value is ``?``.
* **strides**:
 Default value is ``?``.

**Inputs**

Between 2 and 4 inputs.

* **X** (heterogeneous) - **T**:

* **W** (heterogeneous) - **T**:

* **B** (optional, heterogeneous) - **T**:

* **Z** (optional, heterogeneous) - **T**:

**Outputs**

* **Y** (heterogeneous) - **T**:

**Examples**
