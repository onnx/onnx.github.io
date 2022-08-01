
.. _l-onnx-doccom.microsoft-Scale:

=====================
com.microsoft - Scale
=====================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-scale-1:

Scale - 1 (com.microsoft)
=========================

**Version**

* **name**: `Scale (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.Scale>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Scale

**Attributes**

* **scale_down**:
  If true, the output tensor is input tensor devided by scale,
  otherwise, it's input tensor multiplied by scale. The default value
  is false. Default value is ``?``.

**Inputs**

* **input** (heterogeneous) - **T**:
  Input tensor.
* **scale** (heterogeneous) - **ScaleT**:
  Scale scalar tensor.

**Outputs**

* **output** (heterogeneous) - **T**:
  The scaled output tensor.

**Examples**
