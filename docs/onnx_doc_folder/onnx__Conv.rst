
.. _l-onnx-doc-Conv:

====
Conv
====

.. contents::
    :local:


.. _l-onnx-op-conv-11:

Conv - 11
=========

**Version**

* **name**: `Conv (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

The convolution operator consumes an input tensor and a filter, and
computes the output.

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
  present, the dilation defaults is 1 along each spatial axis.
* **group**:
  number of groups input channels and output channels are divided
  into. Default value is ``1``.
* **kernel_shape**:
  The shape of the convolution kernel. If not present, should be
  inferred from input W.
* **pads**:
  Padding for the beginning and ending along each spatial axis, it can
  take any value greater than or equal to 0. The value represent the
  number of pixels added to the beginning and end part of the
  corresponding axis. `pads` format should be as follow [x1_begin,
  x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels
  added at the beginning of axis `i` and xi_end, the number of pixels
  added at the end of axis `i`. This attribute cannot be used
  simultaneously with auto_pad attribute. If not present, the padding
  defaults to 0 along start and end of each spatial axis.
* **strides**:
  Stride along each spatial axis. If not present, the stride defaults
  is 1 along each spatial axis.

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

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.

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

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The convolution operator consumes an input tensor and a filter, and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The convolution operator consumes an input tensor and a filter, and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">computes the output.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">computes the output.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **auto_pad**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **auto_pad**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Where default value is NOTSET, which means explicit padding is used.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Where default value is NOTSET, which means explicit padding is used.</code></td></tr>
    <tr style="1px solid black;"><td><code>8</code></td><td><code>8</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  SAME_UPPER or SAME_LOWER mean pad the input so that <span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span>output</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  SAME_UPPER or SAME_LOWER mean pad the input so that output<span style="color:#196F3D;">_</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">[</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">]</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">9</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  spatial size match the input.In case of odd number add the extra</code></td><td></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">9</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  = ceil(input_shape[i] / strides[i]) for each axis i. The padding</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">10</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  is split between the two sides equally or almost equally (depending</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">11</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  on whether it is even or odd). In case the padding is an odd number,</code></td></tr>
    <tr style="1px solid black;"><td><code>10</code></td><td><code>12</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  padding at the end for SAME_UPPER and at the<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;"> </span>padding <span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span>a<span style="color:#196F3D;">d</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span>t the end for SAME_UPPER and at the</code></td></tr>
    <tr style="1px solid black;"><td><code>11</code></td><td><code>13</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  SAME_LOWER. <span style="color:#BA4A00;">V</span><span style="color:#BA4A00;">A</span><span style="color:#BA4A00;">L</span><span style="color:#BA4A00;">I</span>D<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">m</span>e<span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;">.</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">D</span><span style="color:#BA4A00;">e</span>fault value is 'NOTSET'.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span>SAME_LOWER. Default value is 'NOTSET'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **dilations**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **dilations**:</code></td></tr>
    <tr style="1px solid black;"><td><code>13</code></td><td><code>15</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  dilation value along each spatial axis of the filter.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  dilation value along each spatial axis of the filter.<span style="color:#196F3D;"> </span><span style="color:#196F3D;">I</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">t</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">16</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  present, the dilation defaults is 1 along each spatial axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **group**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **group**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  number of groups input channels and output channels are divided</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  number of groups input channels and output channels are divided</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  into. Default value is 1.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  into. Default value is 1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **kernel_shape**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **kernel_shape**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The shape of the convolution kernel. If not present, should be</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The shape of the convolution kernel. If not present, should be</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  inferred from input W.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  inferred from input W.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **pads**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **pads**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Padding for the beginning and ending along each spatial axis, it can</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Padding for the beginning and ending along each spatial axis, it can</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  take any value greater than or equal to 0. The value represent the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  take any value greater than or equal to 0. The value represent the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  number of pixels added to the beginning and end part of the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  number of pixels added to the beginning and end part of the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding axis. pads format should be as follow [x1_begin,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding axis. pads format should be as follow [x1_begin,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the beginning of axis i and xi_end, the number of pixels</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the beginning of axis i and xi_end, the number of pixels</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the end of axis i. This attribute cannot be used</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the end of axis i. This attribute cannot be used</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  simultaneously with auto_pad attribute. If not present, the padding</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  simultaneously with auto_pad attribute. If not present, the padding</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  defaults to 0 along start and end of each spatial axis.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  defaults to 0 along start and end of each spatial axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **strides**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **strides**:</code></td></tr>
    <tr style="1px solid black;"><td><code>31</code></td><td><code>34</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Stride along each spatial axis.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Stride along each spatial axis.<span style="color:#196F3D;"> </span><span style="color:#196F3D;">I</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">s</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">35</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  is 1 along each spatial axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 3 inputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 3 inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input data tensor from previous layer; has size (N x C x H x W),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input data tensor from previous layer; has size (N x C x H x W),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  where N is the batch size, C is the number of channels, and H and W</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  where N is the batch size, C is the number of channels, and H and W</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  are the height and width. Note that this is for the 2D image.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  are the height and width. Note that this is for the 2D image.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Otherwise the size is (N x C x D1 x D2 ... x Dn). Optionally, if</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Otherwise the size is (N x C x D1 x D2 ... x Dn). Optionally, if</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimension denotation is in effect, the operation expects input data</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimension denotation is in effect, the operation expects input data</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor to arrive with the dimension denotation of [DATA_BATCH,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor to arrive with the dimension denotation of [DATA_BATCH,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **W** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **W** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The weight tensor that will be used in the convolutions; has size (M</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The weight tensor that will be used in the convolutions; has size (M</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x C/group x kH x kW), where C is the number of channels, and kH and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x C/group x kH x kW), where C is the number of channels, and kH and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  kW are the height and width of the kernel, and M is the number of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  kW are the height and width of the kernel, and M is the number of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  feature maps. For more than 2 dimensions, the kernel shape will be</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  feature maps. For more than 2 dimensions, the kernel shape will be</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (M x C/group x k1 x k2 x ... x kn), where (k1 x k2 x ... kn) is the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (M x C/group x k1 x k2 x ... x kn), where (k1 x k2 x ... kn) is the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimension of the kernel. Optionally, if dimension denotation is in</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimension of the kernel. Optionally, if dimension denotation is in</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  effect, the operation expects the weight tensor to arrive with the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  effect, the operation expects the weight tensor to arrive with the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimension denotation of [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimension denotation of [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL,</code></td></tr>
    <tr style="1px solid black;"><td><code>54</code></td><td><code>58</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  FILTER_SPATIAL, FILTER_SPATIAL ...]. <span style="color:#BA4A00;">X</span><span style="color:#BA4A00;">.</span>s<span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">p</span>e<span style="color:#BA4A00;">[</span><span style="color:#BA4A00;">1</span><span style="color:#BA4A00;">]</span> <span style="color:#BA4A00;">=</span><span style="color:#BA4A00;">=</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">(</span><span style="color:#BA4A00;">W</span><span style="color:#BA4A00;">.</span>s<span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">p</span>e<span style="color:#BA4A00;">[</span><span style="color:#BA4A00;">1</span><span style="color:#BA4A00;">]</span> <span style="color:#BA4A00;">*</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  FILTER_SPATIAL, FILTER_SPATIAL ...]. <span style="color:#196F3D;">A</span>s<span style="color:#196F3D;">s</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">z</span>e<span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span> <span style="color:#196F3D;">b</span><span style="color:#196F3D;">a</span>se<span style="color:#196F3D;">d</span> <span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">55</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  group) == C (assuming zero based indices for the shape array). Or in</code></td><td></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">59</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  the shape array, X.shape[1] == (W.shape[1] * group) == C and</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">60</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  W.shape[0] mod G == 0. Or in other words FILTER_IN_CHANNEL</code></td></tr>
    <tr style="1px solid black;"><td><code>56</code></td><td><code>61</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  <span style="color:#BA4A00;">o</span>ther <span style="color:#BA4A00;">w</span>or<span style="color:#BA4A00;">d</span>s <span style="color:#BA4A00;">F</span><span style="color:#BA4A00;">I</span><span style="color:#BA4A00;">L</span><span style="color:#BA4A00;">T</span><span style="color:#BA4A00;">E</span><span style="color:#BA4A00;">R</span><span style="color:#BA4A00;">_</span><span style="color:#BA4A00;">I</span><span style="color:#BA4A00;">N</span><span style="color:#BA4A00;">_</span><span style="color:#BA4A00;">C</span><span style="color:#BA4A00;">H</span><span style="color:#BA4A00;">A</span><span style="color:#BA4A00;">N</span><span style="color:#BA4A00;">N</span><span style="color:#BA4A00;">E</span><span style="color:#BA4A00;">L</span><span style="color:#BA4A00;"> </span>should be equal to DATA_CHANNEL<span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">m</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">l</span>t<span style="color:#196F3D;">i</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span>he<span style="color:#196F3D;"> </span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span>r o<span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">g</span>r<span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">p</span>s should be equal to DATA_CHANNEL</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">62</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  and the number of feature maps M should be a multiple of the number</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">63</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  of groups G.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional 1D bias to be added to the convolution, has size of M.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional 1D bias to be added to the convolution, has size of M.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output data tensor that contains the result of the convolution. The</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output data tensor that contains the result of the convolution. The</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output dimensions are functions of the kernel size, stride size, and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output dimensions are functions of the kernel size, stride size, and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  pad lengths.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  pad lengths.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    </table>

.. _l-onnx-op-conv-1:

Conv - 1
========

**Version**

* **name**: `Conv (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1**.

**Summary**

The convolution operator consumes an input tensor and a filter, and
computes the output.

**Attributes**

* **auto_pad**:
  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.
  Where default value is NOTSET, which means explicit padding is used.
  SAME_UPPER or SAME_LOWER mean pad the input so that the output
  spatial size match the input.In case of odd number add the extra
  padding at the end for SAME_UPPER and at the beginning for
  SAME_LOWER. VALID mean no padding. Default value is ``'NOTSET'``.
* **dilations**:
  dilation value along each spatial axis of the filter.
* **group**:
  number of groups input channels and output channels are divided
  into. Default value is ``1``.
* **kernel_shape**:
  The shape of the convolution kernel. If not present, should be
  inferred from input W.
* **pads**:
  Padding for the beginning and ending along each spatial axis, it can
  take any value greater than or equal to 0. The value represent the
  number of pixels added to the beginning and end part of the
  corresponding axis. `pads` format should be as follow [x1_begin,
  x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels
  added at the beginning of axis `i` and xi_end, the number of pixels
  added at the end of axis `i`. This attribute cannot be used
  simultaneously with auto_pad attribute. If not present, the padding
  defaults to 0 along start and end of each spatial axis.
* **strides**:
  Stride along each spatial axis.

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
  FILTER_SPATIAL, FILTER_SPATIAL ...]. X.shape[1] == (W.shape[1] *
  group) == C (assuming zero based indices for the shape array). Or in
  other words FILTER_IN_CHANNEL should be equal to DATA_CHANNEL.
* **B** (optional, heterogeneous) - **T**:
  Optional 1D bias to be added to the convolution, has size of M.

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output data tensor that contains the result of the convolution. The
  output dimensions are functions of the kernel size, stride size, and
  pad lengths.

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
