
.. _l-onnx-doccom.microsoft-SGDOptimizer:

============================
com.microsoft - SGDOptimizer
============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-sgdoptimizer-1:

SGDOptimizer - 1 (com.microsoft)
================================

**Version**

* **name**: `SGDOptimizer (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.SGDOptimizer>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

**Inputs**

* **ETA** (heterogeneous) - **L**:
  Learning Rate
* **W** (heterogeneous) - **T**:
  Original weight(s)
* **G** (heterogeneous) - **T**:
  Gradient of Weight(s)

**Outputs**

Between 0 and 2 outputs.

* **NW** (optional, heterogeneous) - **T**:
  Updated weight(s)
* **NG** (optional, heterogeneous) - **T**:
  Updated gradients(s)

**Examples**
