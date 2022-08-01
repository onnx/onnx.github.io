
.. _l-onnx-doccom.microsoft-Group:

=====================
com.microsoft - Group
=====================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-group-1:

Group - 1 (com.microsoft)
=========================

**Version**

* **name**: `Group (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.Group>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

if all the inputs are available, the output will be true

**Inputs**

Between 1 and 2147483647 inputs.

* **input_tensors** (variadic) - **T**:
  list of dependency tensors

**Outputs**

* **done** (heterogeneous) - **B**:
  all the dependency tensors are ready

**Examples**
