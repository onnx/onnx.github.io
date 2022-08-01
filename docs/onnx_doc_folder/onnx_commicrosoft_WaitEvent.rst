
.. _l-onnx-doccom.microsoft-WaitEvent:

=========================
com.microsoft - WaitEvent
=========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-waitevent-1:

WaitEvent - 1 (com.microsoft)
=============================

**Version**

* **name**: `WaitEvent (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.WaitEvent>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Wait for an event to be recorded.

**Inputs**

Between 2 and 2147483647 inputs.

* **EventIdentifier** (heterogeneous) - **TInt64**:
  Event identifier to record.
* **InputData** (variadic) - **T**:
  Input data.

**Outputs**

Between 1 and 2147483647 outputs.

* **OutputData** (variadic) - **T**:
  Output data.

**Examples**
