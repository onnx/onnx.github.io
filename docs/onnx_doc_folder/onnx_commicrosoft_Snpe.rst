
.. _l-onnx-doccom.microsoft-Snpe:

====================
com.microsoft - Snpe
====================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-snpe-1:

Snpe - 1 (com.microsoft)
========================

**Version**

* **name**: `Snpe (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.Snpe>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Onnx node for SNPE.

**Attributes**

* **DLC** (required):
  payload of the SNPE DLC file. Default value is ``?``.
* **notes**:
  (Optional) Some notes for the model Default value is ``?``.
* **snpe_version**:
  (Optional) SNPE version used to convert the model. Default value is ``?``.
* **target_device**:
  (Optional) Target device like CPU, DSP, etc. Default value is ``?``.

**Inputs**

Between 1 and 2147483647 inputs.

* **inputs** (variadic, heterogeneous) - **T**:
  List of tensors for SNPE DLC input

**Outputs**

Between 1 and 2147483647 outputs.

* **outputs** (variadic, heterogeneous) - **T**:
  One or more outputs, list of tensors for DLC output

**Examples**
