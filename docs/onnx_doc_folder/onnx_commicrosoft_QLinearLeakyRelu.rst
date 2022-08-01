
.. _l-onnx-doccom.microsoft-QLinearLeakyRelu:

================================
com.microsoft - QLinearLeakyRelu
================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-qlinearleakyrelu-1:

QLinearLeakyRelu - 1 (com.microsoft)
====================================

**Version**

* **name**: `QLinearLeakyRelu (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.QLinearLeakyRelu>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

QLinearLeakyRelu takes quantized input data (Tensor), an argument alpha, and quantize parameter for output,
and produces one output data (Tensor<T>) where the function `f(x) = quantize(alpha * dequantize(x)) for dequantize(x) < 0`,
`f(x) = quantize(dequantize(x)) for dequantize(x) >= 0`, is applied to the data tensor elementwise.

**Attributes**

* **alpha**:
  Coefficient of leakage. Default value is ``?``.

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
