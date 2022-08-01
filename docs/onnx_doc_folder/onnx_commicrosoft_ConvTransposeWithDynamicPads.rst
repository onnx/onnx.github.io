
.. _l-onnx-doccom.microsoft-ConvTransposeWithDynamicPads:

============================================
com.microsoft - ConvTransposeWithDynamicPads
============================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-convtransposewithdynamicpads-1:

ConvTransposeWithDynamicPads - 1 (com.microsoft)
================================================

**Version**

* **name**: `ConvTransposeWithDynamicPads (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.ConvTransposeWithDynamicPads>`_
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
* **dilations**:
 Default value is ``?``.
* **group**:
 Default value is ``?``.
* **kernel_shape**:
 Default value is ``?``.
* **output_padding**:
 Default value is ``?``.
* **strides**:
 Default value is ``?``.

**Inputs**

Between 2 and 4 inputs.

* **X** (heterogeneous) - **T**:

* **W** (heterogeneous) - **T**:

* **Pads** (optional, heterogeneous) - **tensor(int64)**:

* **B** (optional, heterogeneous) - **T**:

**Outputs**

* **Y** (heterogeneous) - **T**:

**Examples**
