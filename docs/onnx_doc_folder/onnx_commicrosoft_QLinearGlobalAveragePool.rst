
.. _l-onnx-doccom.microsoft-QLinearGlobalAveragePool:

========================================
com.microsoft - QLinearGlobalAveragePool
========================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-qlinearglobalaveragepool-1:

QLinearGlobalAveragePool - 1 (com.microsoft)
============================================

**Version**

* **name**: `QLinearGlobalAveragePool (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.QLinearGlobalAveragePool>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

QLinearGlobalAveragePool consumes an input tensor X and applies Average pooling across
the values in the same channel. This is equivalent to AveragePool with kernel size
equal to the spatial dimension of input tensor. Input is of type uint8_t or int8_t.

**Attributes**

* **channels_last**:
 Default value is ``?``.

**Inputs**

* **X** (heterogeneous) - **T**:
  Input data tensor from the previous operator; According to
  channels_last, dimensions for image case are (N x C x H x W), or (N
  x H x W x C) where N is the batch size, C is the number of channels,
  and H and W are the height and the width of the data. For non image
  case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), or
  (N x D1 X D2 ... Dn x C) where N is the batch size.
* **x_scale** (heterogeneous) - **tensor(float)**:
  Scale of quantized input 'X'. It must be a scalar.
* **x_zero_point** (heterogeneous) - **T**:
  Zero point tensor for input 'X'. It must be a scalar.
* **y_scale** (heterogeneous) - **tensor(float)**:
  Scale of quantized output 'Y'. It must be a scalar.
* **y_zero_point** (heterogeneous) - **T**:
  Zero point tensor for output 'Y'. It must be a scalar.

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output data tensor from pooling across the input tensor. The output
  tensor has the same rank as the input. with the N and C value keep
  it value, while the otherdimensions are all 1.

**Examples**
