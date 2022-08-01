
.. _l-onnx-doccom.microsoft-QLinearConv:

===========================
com.microsoft - QLinearConv
===========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-qlinearconv-1:

QLinearConv - 1 (com.microsoft)
===============================

**Version**

* **name**: `QLinearConv (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.QLinearConv>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

**Attributes**

* **auto_pad**:
 Default value is ``?``.
* **channels_last**:
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

Between 8 and 9 inputs.

* **x** (heterogeneous) - **T1**:

* **x_scale** (heterogeneous) - **tensor(float)**:

* **x_zero_point** (heterogeneous) - **T1**:

* **w** (heterogeneous) - **T2**:

* **w_scale** (heterogeneous) - **tensor(float)**:

* **w_zero_point** (heterogeneous) - **T2**:

* **y_scale** (heterogeneous) - **tensor(float)**:

* **y_zero_point** (heterogeneous) - **T3**:

* **B** (optional, heterogeneous) - **T4**:

**Outputs**

* **y** (heterogeneous) - **T3**:

**Examples**
