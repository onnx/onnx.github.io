
.. _l-onnx-doccom.microsoft-BitmaskBiasDropout:

==================================
com.microsoft - BitmaskBiasDropout
==================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-bitmaskbiasdropout-1:

BitmaskBiasDropout - 1 (com.microsoft)
======================================

**Version**

* **name**: `BitmaskBiasDropout (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.BitmaskBiasDropout>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

output, dropout_bitmask = Dropout(data + bias, ratio) + residual, Intended to specialize the dropout pattern commonly found in transformer models.

**Attributes**

* **seed**:
  (Optional) Seed to the random generator, if not specified we will
  auto generate one. Default value is ``?``.

**Inputs**

Between 2 and 5 inputs.

* **data** (heterogeneous) - **T**:
  The input data as Tensor.
* **bias** (heterogeneous) - **T**:
  The bias input, a vector with the same shape as last dim of data OR
  same shape with data
* **residual** (optional, heterogeneous) - **T**:
  The residual input, must have the same shape as data
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
  The output mask of dropout.

**Examples**
