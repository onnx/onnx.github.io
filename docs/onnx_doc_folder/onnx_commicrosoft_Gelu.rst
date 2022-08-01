
.. _l-onnx-doccom.microsoft-Gelu:

====================
com.microsoft - Gelu
====================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-gelu-1:

Gelu - 1 (com.microsoft)
========================

**Version**

* **name**: `Gelu (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.Gelu>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Gaussian Error Linear Unit.
A high-performing neural network activation function.The GELU nonlinearity is
the expected transformation of a stochastic regularizer which randomly applies
the identity or zero map to a neuron's input. The GELU nonlinearity weights
inputs by their magnitude, rather than gates inputs by their sign as in ReLUs.

**Inputs**

* **X** (heterogeneous) - **T**:
  The input data as Tensor.

**Outputs**

* **Y** (heterogeneous) - **T**:
  The output.

**Examples**
