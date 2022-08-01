
.. _l-onnx-doccom.microsoft-QuantizeLinear:

==============================
com.microsoft - QuantizeLinear
==============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-quantizelinear-1:

QuantizeLinear - 1 (com.microsoft)
==================================

**Version**

* **name**: `QuantizeLinear (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.QuantizeLinear>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

The linear quantization operator. It consumes a full precision data, a scale, a zero point to compute the low precision / quantized tensor.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point).For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
Scale and zero point must have same shape. They must be either scalar (per tensor) or 1-D tensor (per 'axis').

**Attributes**

* **axis**:
  The axis along which same quantization parameters are applied. It's
  optional.If it's not specified, it means per-tensor quantization and
  input 'x_scale' and 'x_zero_point' must be scalars.If it's
  specified, it means per 'axis' quantization and input 'x_scale' and
  'x_zero_point' must be 1-D tensors. Default value is ``?``.

**Inputs**

* **x** (heterogeneous) - **T1**:
  N-D full precision Input tensor to be quantized.
* **y_scale** (heterogeneous) - **T1**:
  Scale for doing quantization to get 'y'. It could be a scalar or a
  1-D tensor,which means a per-tensor or per-axis quantization. If
  it's a 1-D tensor, its number of elements should be equal to the
  dimension value of 'axis' dimension of input 'x'.
* **y_zero_point** (heterogeneous) - **T2**:
  Zero point for doing quantization to get 'y'. It could be a scalar
  or a 1-D tensor, which means a per-tensoror per-axis quantization.
  If it's a 1-D tensor, its number of elements should be equal to the
  dimension value of 'axis' dimension of input 'x'.

**Outputs**

* **y** (heterogeneous) - **T2**:
  N-D quantized output tensor. It has same shape as input 'x'.

**Examples**

**axis**

::

    node = onnx.helper.make_node('QuantizeLinear',
                                 inputs=['x', 'y_scale', 'y_zero_point'],
                                 outputs=['y'],)

    x = np.array([[[[-162, 10],
                    [-100, 232],
                    [-20, -50]],

                   [[-76, 0],
                    [0, 252],
                    [32, -44]],

                   [[245, -485],
                    [-960, -270],
                    [-375, -470]], ], ], dtype=np.float32)
    y_scale = np.array([2, 4, 5], dtype=np.float32)
    y_zero_point = np.array([84, 24, 196], dtype=np.uint8)
    y = (x / y_scale.reshape(1, 3, 1, 1) + y_zero_point.reshape(1, 3, 1, 1)).astype(np.uint8)

    expect(node, inputs=[x, y_scale, y_zero_point], outputs=[y],
           name='test_quantizelinear_axis')
