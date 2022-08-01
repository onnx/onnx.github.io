
.. _l-onnx-doccom.microsoft-SliceGrad:

=========================
com.microsoft - SliceGrad
=========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-slicegrad-1:

SliceGrad - 1 (com.microsoft)
=============================

**Version**

* **name**: `SliceGrad (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.SliceGrad>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

**Inputs**

Between 4 and 6 inputs.

* **dY** (heterogeneous) - **T**:
  Gradient of output
* **shape** (heterogeneous) - **I**:
  Shape of the Slice input X.
* **starts** (heterogeneous) - **Tind**:
  Tensor of starting indices of corresponding axis in axes
* **ends** (heterogeneous) - **Tind**:
  Tensor of starting indices of corresponding axis in 'axes'
* **axes** (optional, heterogeneous) - **Tind**:
  Tensor of axes that `starts` and `ends` apply to
* **steps** (optional, heterogeneous) - **Tind**:
  Tensor of slice step of corresponding axis in `axes`

**Outputs**

* **dX** (heterogeneous) - **T**:
  Gradient of input

**Examples**
