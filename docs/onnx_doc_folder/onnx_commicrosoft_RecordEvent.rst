
.. _l-onnx-doccom.microsoft-RecordEvent:

===========================
com.microsoft - RecordEvent
===========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-recordevent-1:

RecordEvent - 1 (com.microsoft)
===============================

**Version**

* **name**: `RecordEvent (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.RecordEvent>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Record an event.

**Inputs**

Between 2 and 2147483647 inputs.

* **EventIdentifier** (heterogeneous) - **TInt64**:
  Event identifier to record.
* **InputData** (variadic) - **T**:
  Input data.

**Outputs**

Between 0 and 2147483647 outputs.

* **OutputData** (variadic) - **T**:
  Output data.

**Examples**
