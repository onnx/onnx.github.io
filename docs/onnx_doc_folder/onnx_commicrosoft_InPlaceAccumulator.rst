
.. _l-onnx-doccom.microsoft-InPlaceAccumulator:

==================================
com.microsoft - InPlaceAccumulator
==================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-inplaceaccumulator-1:

InPlaceAccumulator - 1 (com.microsoft)
======================================

**Version**

* **name**: `InPlaceAccumulator (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.InPlaceAccumulator>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

in-place accumulator for tensors

**Inputs**

Between 2 and 3 inputs.

* **old_sum** (heterogeneous) - **T**:
  historical result of accumulator
* **value** (heterogeneous) - **T_GRAD**:
  the value that will be added to the accumulator
* **update_signal** (optional, heterogeneous) - **T_BOOL**:
  This signal indicates if tensor should be updated

**Outputs**

* **new_sum** (heterogeneous) - **T**:
  updated result of accumulator

**Examples**
