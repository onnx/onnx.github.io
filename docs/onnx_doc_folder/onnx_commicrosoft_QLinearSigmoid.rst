
.. _l-onnx-doccom.microsoft-QLinearSigmoid:

==============================
com.microsoft - QLinearSigmoid
==============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-qlinearsigmoid-1:

QLinearSigmoid - 1 (com.microsoft)
==================================

**Version**

* **name**: `QLinearSigmoid (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.QLinearSigmoid>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

QLinearSigmoid takes quantized input data (Tensor), and quantize parameter for output, and produces one output data
(Tensor<T>) where the function `f(x) = quantize(Sigmoid(dequantize(x)))`, is applied to the data tensor elementwise.
Wwhere the function `Sigmoid(x) = 1 / (1 + exp(-x))`

**Inputs**

Between 4 and 5 inputs.

* **X** (heterogeneous) - **T**:
  Input tensor
* **X_scale** (heterogeneous) - **tensor(float)**:
  Input X's scale. It's a scalar, which means a per-tensor/layer
  quantization.
* **X_zero_point** (optional, heterogeneous) - **T**:
  Input X's zero point. Default value is 0 if it's not specified. It's
  a scalar, which means a per-tensor/layer quantization.
* **Y_scale** (heterogeneous) - **tensor(float)**:
  Output Y's scale. It's a scalar, which means a per-tensor/layer
  quantization.
* **Y_zero_point** (optional, heterogeneous) - **T**:
  Output Y's zero point. Default value is 0 if it's not specified.
  It's a scalar, which means a per-tensor/layer quantization.

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output tensor

**Examples**
