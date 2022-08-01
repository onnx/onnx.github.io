
.. _l-onnx-doccom.microsoft-DequantizeLinear:

================================
com.microsoft - DequantizeLinear
================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-dequantizelinear-1:

DequantizeLinear - 1 (com.microsoft)
====================================

**Version**

* **name**: `DequantizeLinear (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.DequantizeLinear>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

The linear dequantization operator. It consumes a quantized data, a scale, a zero point and computes the full precision data.
The dequantization formula is y = (x - x_zero_point) * x_scale.
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
  N-D quantized Input tensor to be de-quantized.
* **x_scale** (heterogeneous) - **T2**:
  Scale for input 'x'. It could be a scalar or a 1-D tensor, which
  means a per-tensor or per-axis quantization.If it's a 1-D tensor,
  its number of elements should be equal to the dimension value of
  'axis' dimension of input 'x'.
* **x_zero_point** (heterogeneous) - **T1**:
  Zero point for input 'x'. It could be a scalar or a 1-D tensor,
  which means a per-tensor or per-axis quantization.If it's a 1-D
  tensor, its number of elements should be equal to the dimension
  value of 'axis' dimension of input 'x'.

**Outputs**

* **y** (heterogeneous) - **T2**:
  N-D full precision output tensor. It has same shape as input 'x'.

**Examples**

**axis**

::

    node = onnx.helper.make_node('DequantizeLinear',
                                 inputs=['x', 'x_scale', 'x_zero_point'],
                                 outputs=['y'],)

    # 1-D tensor zero point and scale of size equal to axis 1 of the input tensor
    x = np.array([[[[3, 89],
                    [34, 200],
                    [74, 59]],

                   [[5, 24],
                    [24, 87],
                    [32, 13]],

                   [[245, 99],
                    [4, 142],
                    [121, 102]], ], ], dtype=np.uint8)
    x_scale = np.array([2, 4, 5], dtype=np.float32)
    x_zero_point = np.array([84, 24, 196], dtype=np.uint8)
    y = (x.astype(np.float32) - x_zero_point.reshape(1, 3, 1, 1).astype(np.float32)) * x_scale.reshape(1, 3, 1, 1)

    expect(node, inputs=[x, x_scale, x_zero_point], outputs=[y],
           name='test_dequantizelinear_axis')
