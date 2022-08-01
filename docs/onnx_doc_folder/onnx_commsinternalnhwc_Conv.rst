
.. _l-onnx-doccom.ms.internal.nhwc-Conv:

===========================
com.ms.internal.nhwc - Conv
===========================

.. contents::
    :local:


.. _l-onnx-opcom-ms-internal-nhwc-conv-11:

Conv - 11 (com.ms.internal.nhwc)
================================

**Version**

* **name**: `Conv (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.ms.internal.nhwc.Conv>`_
* **domain**: **com.ms.internal.nhwc**
* **since_version**: **11**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 11 of domain com.ms.internal.nhwc**.

**Summary**

The convolution operator consumes an input tensor and a filter, and
computes the output.

**Attributes**

* **activation**:
 Default value is ``?``.
* **activation_params**:
 Default value is ``?``.
* **auto_pad**:
  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.
  Where default value is NOTSET, which means explicit padding is used.
  SAME_UPPER or SAME_LOWER mean pad the input so that `output_shape[i]
  = ceil(input_shape[i] / strides[i])` for each axis `i`. The padding
  is split between the two sides equally or almost equally (depending
  on whether it is even or odd). In case the padding is an odd number,
  the extra padding is added at the end for SAME_UPPER and at the
  beginning for SAME_LOWER. Default value is ``?``.
* **dilations**:
  dilation value along each spatial axis of the filter. If not
  present, the dilation defaults is 1 along each spatial axis. Default value is ``?``.
* **group**:
  number of groups input channels and output channels are divided
  into. Default value is ``?``.
* **kernel_shape**:
  The shape of the convolution kernel. If not present, should be
  inferred from input W. Default value is ``?``.
* **pads**:
  Padding for the beginning and ending along each spatial axis, it can
  take any value greater than or equal to 0. The value represent the
  number of pixels added to the beginning and end part of the
  corresponding axis. `pads` format should be as follow [x1_begin,
  x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels
  added at the beginning of axis `i` and xi_end, the number of pixels
  added at the end of axis `i`. This attribute cannot be used
  simultaneously with auto_pad attribute. If not present, the padding
  defaults to 0 along start and end of each spatial axis. Default value is ``?``.
* **strides**:
  Stride along each spatial axis. If not present, the stride defaults
  is 1 along each spatial axis. Default value is ``?``.

**Inputs**

Between 2 and 3 inputs.

* **X** (heterogeneous) - **T**:
  Input data tensor from previous layer; has size (N x C x H x W),
  where N is the batch size, C is the number of channels, and H and W
  are the height and width. Note that this is for the 2D image.
  Otherwise the size is (N x C x D1 x D2 ... x Dn). Optionally, if
  dimension denotation is in effect, the operation expects input data
  tensor to arrive with the dimension denotation of [DATA_BATCH,
  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].
* **W** (heterogeneous) - **T**:
  The weight tensor that will be used in the convolutions; has size (M
  x C/group x kH x kW), where C is the number of channels, and kH and
  kW are the height and width of the kernel, and M is the number of
  feature maps. For more than 2 dimensions, the kernel shape will be
  (M x C/group x k1 x k2 x ... x kn), where (k1 x k2 x ... kn) is the
  dimension of the kernel. Optionally, if dimension denotation is in
  effect, the operation expects the weight tensor to arrive with the
  dimension denotation of [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL,
  FILTER_SPATIAL, FILTER_SPATIAL ...]. Assuming zero based indices for
  the shape array, X.shape[1] == (W.shape[1] * group) == C and
  W.shape[0] mod G == 0. Or in other words FILTER_IN_CHANNEL
  multiplied by the number of groups should be equal to DATA_CHANNEL
  and the number of feature maps M should be a multiple of the number
  of groups G.
* **B** (optional, heterogeneous) - **T**:
  Optional 1D bias to be added to the convolution, has size of M.

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output data tensor that contains the result of the convolution. The
  output dimensions are functions of the kernel size, stride size, and
  pad lengths.

**Examples**

**conv_with_strides**

::

    x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 7, 5) input tensor
                    [5., 6., 7., 8., 9.],
                    [10., 11., 12., 13., 14.],
                    [15., 16., 17., 18., 19.],
                    [20., 21., 22., 23., 24.],
                    [25., 26., 27., 28., 29.],
                    [30., 31., 32., 33., 34.]]]]).astype(np.float32)
    W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                    [1., 1., 1.],
                    [1., 1., 1.]]]]).astype(np.float32)

    # Convolution with strides=2 and padding
    node_with_padding = onnx.helper.make_node(
        'Conv',
        inputs=['x', 'W'],
        outputs=['y'],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[2, 2],  # Default values for other attributes: dilations=[1, 1], groups=1
    )
    y_with_padding = np.array([[[[12., 27., 24.],  # (1, 1, 4, 3) output tensor
                                 [63., 108., 81.],
                                 [123., 198., 141.],
                                 [112., 177., 124.]]]]).astype(np.float32)
    expect(node_with_padding, inputs=[x, W], outputs=[y_with_padding],
           name='test_conv_with_strides_padding')

    # Convolution with strides=2 and no padding
    node_without_padding = onnx.helper.make_node(
        'Conv',
        inputs=['x', 'W'],
        outputs=['y'],
        kernel_shape=[3, 3],
        pads=[0, 0, 0, 0],
        strides=[2, 2],  # Default values for other attributes: dilations=[1, 1], groups=1
    )
    y_without_padding = np.array([[[[54., 72.],  # (1, 1, 3, 2) output tensor
                                    [144., 162.],
                                    [234., 252.]]]]).astype(np.float32)
    expect(node_without_padding, inputs=[x, W], outputs=[y_without_padding],
           name='test_conv_with_strides_no_padding')

    # Convolution with strides=2 and padding only along one dimension (the H dimension in NxCxHxW tensor)
    node_with_asymmetric_padding = onnx.helper.make_node(
        'Conv',
        inputs=['x', 'W'],
        outputs=['y'],
        kernel_shape=[3, 3],
        pads=[1, 0, 1, 0],
        strides=[2, 2],  # Default values for other attributes: dilations=[1, 1], groups=1
    )
    y_with_asymmetric_padding = np.array([[[[21., 33.],  # (1, 1, 4, 2) output tensor
                                            [99., 117.],
                                            [189., 207.],
                                            [171., 183.]]]]).astype(np.float32)
    expect(node_with_asymmetric_padding, inputs=[x, W], outputs=[y_with_asymmetric_padding],
           name='test_conv_with_strides_and_asymmetric_padding')

**conv_with_autopad_same**

::

    x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                    [5., 6., 7., 8., 9.],
                    [10., 11., 12., 13., 14.],
                    [15., 16., 17., 18., 19.],
                    [20., 21., 22., 23., 24.]]]]).astype(np.float32)
    W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                    [1., 1., 1.],
                    [1., 1., 1.]]]]).astype(np.float32)

    # Convolution with auto_pad='SAME_LOWER' and strides=2
    node = onnx.helper.make_node(
        'Conv',
        inputs=['x', 'W'],
        outputs=['y'],
        auto_pad='SAME_LOWER',
        kernel_shape=[3, 3],
        strides=[2, 2],
    )
    y = np.array([[[[12., 27., 24.],
                 [63., 108., 81.],
                 [72., 117., 84.]]]]).astype(np.float32)
    expect(node, inputs=[x, W], outputs=[y],
           name='test_conv_with_autopad_same')
