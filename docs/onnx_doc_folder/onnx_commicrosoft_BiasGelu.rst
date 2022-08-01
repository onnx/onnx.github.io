
.. _l-onnx-doccom.microsoft-BiasGelu:

========================
com.microsoft - BiasGelu
========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-biasgelu-1:

BiasGelu - 1 (com.microsoft)
============================

**Version**

* **name**: `BiasGelu (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.BiasGelu>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Bias Gelu.
It's an extension of Gelu. It takes the sum of input A and bias input B as the input of Gelu activation.

**Inputs**

* **A** (heterogeneous) - **T**:
  The normal input data.
* **B** (heterogeneous) - **T**:
  The bias input data that is a 1D tensor.

**Outputs**

* **C** (heterogeneous) - **T**:
  The output.

**Examples**
