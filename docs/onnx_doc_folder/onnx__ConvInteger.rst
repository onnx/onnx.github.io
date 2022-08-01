
.. _l-onnx-doc-ConvInteger:

===========
ConvInteger
===========

.. contents::
    :local:


.. _l-onnx-op-convinteger-10:

ConvInteger - 10
================

**Version**

* **name**: `ConvInteger (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvInteger>`_
* **domain**: **main**
* **since_version**: **10**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 10**.

**Summary**

The integer convolution operator consumes an input tensor, its zero-point, a filter, and its zero-point,
and computes the output. The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.

**Attributes**

* **auto_pad**:
  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.
  Where default value is NOTSET, which means explicit padding is used.
  SAME_UPPER or SAME_LOWER mean pad the input so that `output_shape[i]
  = ceil(input_shape[i] / strides[i])` for each axis `i`. The padding
  is split between the two sides equally or almost equally (depending
  on whether it is even or odd). In case the padding is an odd number,
  the extra padding is added at the end for SAME_UPPER and at the
  beginning for SAME_LOWER. Default value is ``'NOTSET'``.
* **dilations**:
  dilation value along each spatial axis of the filter. If not
  present, the dilation defaults to 1 along each axis.
* **group**:
  number of groups input channels and output channels are divided
  into. default is 1. Default value is ``1``.
* **kernel_shape**:
  The shape of the convolution kernel. If not present, should be
  inferred from input 'w'.
* **pads**:
  Padding for the beginning and ending along each spatial axis, it can
  take any value greater than or equal to 0.The value represent the
  number of pixels added to the beginning and end part of the
  corresponding axis.`pads` format should be as follow [x1_begin,
  x2_begin...x1_end, x2_end,...], where xi_begin the number ofpixels
  added at the beginning of axis `i` and xi_end, the number of pixels
  added at the end of axis `i`.This attribute cannot be used
  simultaneously with auto_pad attribute. If not present, the padding
  defaultsto 0 along start and end of each spatial axis.
* **strides**:
  Stride along each spatial axis. If not present, the stride defaults
  to 1 along each axis.

**Inputs**

Between 2 and 4 inputs.

* **x** (heterogeneous) - **T1**:
  Input data tensor from previous layer; has size (N x C x H x W),
  where N is the batch size, C is the number of channels, and H and W
  are the height and width. Note that this is for the 2D image.
  Otherwise the size is (N x C x D1 x D2 ... x Dn). Optionally, if
  dimension denotation is in effect, the operation expects input data
  tensor to arrive with the dimension denotation of [DATA_BATCH,
  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].
* **w** (heterogeneous) - **T2**:
  The weight tensor that will be used in the convolutions; has size (M
  x C/group x kH x kW), where C is the number of channels, and kH and
  kW are the height and width of the kernel, and M is the number of
  feature maps. For more than 2 dimensions, the kernel shape will be
  (M x C/group x k1 x k2 x ... x kn), where (k1 x k2 x ... kn) is the
  dimension of the kernel. Optionally, if dimension denotation is in
  effect, the operation expects the weight tensor to arrive with the
  dimension denotation of [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL,
  FILTER_SPATIAL, FILTER_SPATIAL ...]. X.shape[1] == (W.shape[1] *
  group) == C (assuming zero based indices for the shape array). Or in
  other words FILTER_IN_CHANNEL should be equal to DATA_CHANNEL.
* **x_zero_point** (optional, heterogeneous) - **T1**:
  Zero point tensor for input 'x'. It's optional and default value is
  0. It's a scalar, which means a per-tensor/layer quantization.
* **w_zero_point** (optional, heterogeneous) - **T2**:
  Zero point tensor for input 'w'. It's optional and default value is
  0.  It could be a scalar or a 1-D tensor, which means a per-
  tensor/layer or per output channel quantization. If it's a 1-D
  tensor, its number of elements should be equal to the number of
  output channels (M)

**Outputs**

* **y** (heterogeneous) - **T3**:
  Output data tensor that contains the result of the convolution. The
  output dimensions are functions of the kernel size, stride size, and
  pad lengths.

**Type Constraints**

* **T1** in (
  tensor(int8),
  tensor(uint8)
  ):
  Constrain input x and its zero point data type to 8-bit integer
  tensor.
* **T2** in (
  tensor(int8),
  tensor(uint8)
  ):
  Constrain input w and its zero point data type to 8-bit integer
  tensor.
* **T3** in (
  tensor(int32)
  ):
  Constrain output y data type to 32-bit integer tensor.

**Examples**

**without_padding**

::

    x = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.uint8).reshape((1, 1, 3, 3))
    x_zero_point = np.uint8(1)
    w = np.array([1, 1, 1, 1]).astype(np.uint8).reshape((1, 1, 2, 2))

    y = np.array([12, 16, 24, 28]).astype(np.int32).reshape(1, 1, 2, 2)

    # ConvInteger without padding
    convinteger_node = onnx.helper.make_node('ConvInteger',
        inputs=['x', 'w', 'x_zero_point'],
        outputs=['y'])

    expect(convinteger_node, inputs=[x, w, x_zero_point], outputs=[y],
           name='test_convinteger_without_padding')

**with_padding**

::

    x = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.uint8).reshape((1, 1, 3, 3))
    x_zero_point = np.uint8(1)
    w = np.array([1, 1, 1, 1]).astype(np.uint8).reshape((1, 1, 2, 2))

    y = np.array([1, 3, 5, 3, 5, 12, 16, 9, 11, 24, 28, 15, 7, 15, 17, 9]).astype(np.int32).reshape((1, 1, 4, 4))

    # ConvInteger with padding
    convinteger_node_with_padding = onnx.helper.make_node('ConvInteger',
        inputs=['x', 'w', 'x_zero_point'],
        outputs=['y'],
        pads=[1, 1, 1, 1],)

    expect(convinteger_node_with_padding, inputs=[x, w, x_zero_point], outputs=[y],
           name='test_convinteger_with_padding')
