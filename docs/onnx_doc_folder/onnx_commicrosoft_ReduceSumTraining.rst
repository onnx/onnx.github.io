
.. _l-onnx-doccom.microsoft-ReduceSumTraining:

=================================
com.microsoft - ReduceSumTraining
=================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-reducesumtraining-1:

ReduceSumTraining - 1 (com.microsoft)
=====================================

**Version**

* **name**: `ReduceSumTraining (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.ReduceSumTraining>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

ReduceSumTraining

**Attributes**

* **keepdims**:
  Keep the reduced dimension or not, default 1 mean keep reduced
  dimension. Default value is ``?``.
* **noop_with_empty_axes**:
  Perform reduction or not when axes is empty, default false mean
  perform reduction.when axes is empty and this attribute is set to
  true, input tensor will not be reduced,thus output tensor would be
  equivalent to input tensor. Default value is ``?``.

**Inputs**

* **data** (heterogeneous) - **T**:
  An input tensor.
* **axes** (heterogeneous) - **tensor(int64)**:
  A list of integers, along which to reduce. The default is to reduce
  over all the dimensions of the input tensor. Accepted range is [-r,
  r-1] where r = rank(data).

**Outputs**

* **reduced** (heterogeneous) - **T**:
  Reduced output tensor.

**Examples**
