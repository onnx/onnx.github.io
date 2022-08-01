
.. _l-onnx-doccom.microsoft-Recv:

====================
com.microsoft - Recv
====================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-recv-1:

Recv - 1 (com.microsoft)
========================

**Version**

* **name**: `Recv (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.Recv>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Receive a tensor from the the specified source.

**Attributes**

* **element_types** (required):
  Element types of the received tensors. Default value is ``?``.
* **tag** (required):
  The tag of the message carrying Data. Default value is ``?``.

**Inputs**

* **InputSignal** (heterogeneous) - **TBool**:
  Input control signal. It must be a scalar.
* **Remote** (heterogeneous) - **TInt64**:
  Remote src rank. It must be a scalar.

**Outputs**

Between 2 and 2147483647 outputs.

* **OutputSignal** (heterogeneous) - **TBool**:
  Output control signal. It must be a scalar.
* **Data** (variadic) - **V**:
  The Received tensors.

**Examples**
