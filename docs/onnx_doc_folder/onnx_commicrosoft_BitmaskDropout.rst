
.. _l-onnx-doccom.microsoft-BitmaskDropout:

==============================
com.microsoft - BitmaskDropout
==============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-bitmaskdropout-1:

BitmaskDropout - 1 (com.microsoft)
==================================

**Version**

* **name**: `BitmaskDropout (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.BitmaskDropout>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

BitmaskDropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar).
It produces two tensor outputs: output (floating-point tensor) and mask (optional `Tensor<uint32>`). If `training_mode` is true then the output Y will be a random dropout.
Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode, the user can simply not pass `training_mode` input or set it to false.
::

    output = scale * data * mask,

where
::

    scale = 1. / (1. - ratio).

This op functions in much the same was as Dropout-11 and Dropout-13 do, execpt that the mask is output as a bit-packed uint32 tensor, instead of a boolean tensor.

**Attributes**

* **seed**:
  (Optional) Seed to the random generator, if not specified we will
  auto generate one. Default value is ``?``.

**Inputs**

Between 1 and 3 inputs.

* **data** (heterogeneous) - **T**:
  The input data as Tensor.
* **ratio** (optional, heterogeneous) - **T1**:
  The ratio of random dropout, with value in [0, 1). If this input was
  not set, or if it was set to 0, the output would be a simple copy of
  the input. If it's non-zero, output will be a random dropout of the
  scaled input, which is typically the case during training. It is an
  optional value, if not specified it will default to 0.5.
* **training_mode** (optional, heterogeneous) - **T2**:
  If set to true then it indicates dropout is being used for training.
  It is an optional value hence unless specified explicitly, it is
  false. If it is false, ratio is ignored and the operation mimics
  inference mode where nothing will be dropped from the input data and
  if mask is requested as output it will contain all ones.

**Outputs**

Between 1 and 2 outputs.

* **output** (heterogeneous) - **T**:
  The output.
* **mask** (optional, heterogeneous) - **T3**:
  The bit-packed output mask.

**Examples**
