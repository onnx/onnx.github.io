
.. _l-onnx-doc-ConvTranspose:

=============
ConvTranspose
=============

.. contents::
    :local:


.. _l-onnx-op-convtranspose-11:

ConvTranspose - 11
==================

**Version**

* **name**: `ConvTranspose (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

The convolution transpose operator consumes an input tensor and a filter,
and computes the output.

If the pads parameter is provided the shape of the output is calculated via the following equation:

  output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]

output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

  total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
  If (auto_pads == SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
  Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).

**Attributes**

* **auto_pad**:
  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.
  Where default value is NOTSET, which means explicit padding is used.
  SAME_UPPER or SAME_LOWER mean pad the input so that `output_shape[i]
  = input_shape[i] * strides[i]` for each axis `i`. The padding is
  split between the two sides equally or almost equally (depending on
  whether it is even or odd). In case the padding is an odd number,
  the extra padding is added at the end for SAME_UPPER and at the
  beginning for SAME_LOWER. Default value is ``'NOTSET'``.
* **dilations**:
  dilation value along each spatial axis of the filter. If not
  present, the dilation defaults to 1 along each spatial axis.
* **group**:
  number of groups input channels and output channels are divided
  into. Default value is ``1``.
* **kernel_shape**:
  The shape of the convolution kernel. If not present, should be
  inferred from input W.
* **output_padding**:
  Additional elements added to the side with higher coordinate indices
  in the output. Each padding value in "output_padding" must be less
  than the corresponding stride/dilation dimension. By default, this
  attribute is a zero vector. Note that this attribute doesn't
  directly affect the computed output values. It only controls the
  selection of the computed values, so changing this attribute only
  adds or removes output elements. If "output_shape" is explicitly
  provided, "output_padding" does not contribute additional size to
  "output_shape" but participates in the computation of the needed
  padding amount. This is also called adjs or adjustment in some
  frameworks.
* **output_shape**:
  The shape of the output can be explicitly set which will cause pads
  values to be auto generated. If output_shape is specified pads
  values are ignored. See doc for details for equations to generate
  pads
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
  to 1 along each spatial axis.

**Inputs**

Between 2 and 3 inputs.

* **X** (heterogeneous) - **T**:
  Input data tensor from previous layer; has size (N x C x H x W),
  where N is the batch size, C is the number of channels, and H and W
  are the height and width. Note that this is for the 2D image.
  Otherwise the size is (N x C x D1 x D2 ... x Dn)
* **W** (heterogeneous) - **T**:
  The weight tensor that will be used in the convolutions; has size (C
  x M/group x kH x kW), where C is the number of channels, and kH and
  kW are the height and width of the kernel, and M is the number of
  feature maps. For more than 2 dimensions, the weight shape will be
  (C x M/group x k1 x k2 x ... x kn), where (k1 x k2 x ... x kn) is
  the dimension of the kernel. The number of channels in the output
  should be equal to W.shape[1] * group (assuming zero based indices
  of the shape array)
* **B** (optional, heterogeneous) - **T**:
  Optional 1D bias to be added to the convolution, has size of M.

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output data tensor that contains the result of the convolution. The
  output dimensions are functions of the kernel size, stride size, pad
  lengths and group count. The number of channels in the output should
  be equal to W.shape[1] * group (assuming zero based indices of the
  shape array)

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.

**Examples**

**convtranspose_1d**

::

    x = np.array([[[0., 1., 2.]]]).astype(np.float32)  # (1, 1, 3)

    W = np.array([[[1., 1., 1.],  # (1, 2, 3)
                   [1., 1., 1.]]]).astype(np.float32)

    node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"])

    y = np.array([[[0., 1., 3., 3., 2.],  # (1, 2, 5)
                   [0., 1., 3., 3., 2.]]]).astype(np.float32)

    expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_1d')

**convtranspose_3d**

::

    x = np.array([[[[[0., 1., 2., 3., 4.],  # (1, 1, 3, 4, 5)
                     [5., 6., 7., 8., 9.],
                     [10., 11., 12., 13., 14.],
                     [15., 16., 17., 18., 19.]],
                    [[20., 21., 22., 23., 24.],
                     [25., 26., 27., 28., 29.],
                     [30., 31., 32., 33., 34.],
                     [35., 36., 37., 38., 39.]],
                    [[40., 41., 42., 43., 44.],
                     [45., 46., 47., 48., 49.],
                     [50., 51., 52., 53., 54.],
                     [55., 56., 57., 58., 59.]]]]]).astype(np.float32)

    W = np.array([[[[[1., 1., 1.],  # (1, 2, 3, 3, 3)
                     [1., 1., 1.],
                     [1., 1., 1.]],
                    [[1., 1., 1.],
                     [1., 1., 1.],
                     [1., 1., 1.]],
                    [[1., 1., 1.],
                     [1., 1., 1.],
                     [1., 1., 1.]]],
                   [[[1., 1., 1.],
                     [1., 1., 1.],
                     [1., 1., 1.]],
                    [[1., 1., 1.],
                     [1., 1., 1.],
                     [1., 1., 1.]],
                    [[1., 1., 1.],
                     [1., 1., 1.],
                     [1., 1., 1.]]]]]).astype(np.float32)

    node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"])

    y = np.array([[[[[0., 1., 3., 6., 9., 7., 4.],  # (1, 2, 5, 6, 7)
                     [5., 12., 21., 27., 33., 24., 13.],
                     [15., 33., 54., 63., 72., 51., 27.],
                     [30., 63., 99., 108., 117., 81., 42.],
                     [25., 52., 81., 87., 93., 64., 33.],
                     [15., 31., 48., 51., 54., 37., 19.]],

                    [[20., 42., 66., 72., 78., 54., 28.],
                     [50., 104., 162., 174., 186., 128., 66.],
                     [90., 186., 288., 306., 324., 222., 114.],
                     [120., 246., 378., 396., 414., 282., 144.],
                     [90., 184., 282., 294., 306., 208., 106.],
                     [50., 102., 156., 162., 168., 114., 58.]],

                    [[60., 123., 189., 198., 207., 141., 72.],
                     [135., 276., 423., 441., 459., 312., 159.],
                     [225., 459., 702., 729., 756., 513., 261.],
                     [270., 549., 837., 864., 891., 603., 306.],
                     [195., 396., 603., 621., 639., 432., 219.],
                     [105., 213., 324., 333., 342., 231., 117.]],

                    [[60., 122., 186., 192., 198., 134., 68.],
                     [130., 264., 402., 414., 426., 288., 146.],
                     [210., 426., 648., 666., 684., 462., 234.],
                     [240., 486., 738., 756., 774., 522., 264.],
                     [170., 344., 522., 534., 546., 368., 186.],
                     [90., 182., 276., 282., 288., 194., 98.]],

                    [[40., 81., 123., 126., 129., 87., 44.],
                     [85., 172., 261., 267., 273., 184., 93.],
                     [135., 273., 414., 423., 432., 291., 147.],
                     [150., 303., 459., 468., 477., 321., 162.],
                     [105., 212., 321., 327., 333., 224., 113.],
                     [55., 111., 168., 171., 174., 117., 59.]]],

                   [[[0., 1., 3., 6., 9., 7., 4.],
                     [5., 12., 21., 27., 33., 24., 13.],
                     [15., 33., 54., 63., 72., 51., 27.],
                     [30., 63., 99., 108., 117., 81., 42.],
                     [25., 52., 81., 87., 93., 64., 33.],
                     [15., 31., 48., 51., 54., 37., 19.]],

                    [[20., 42., 66., 72., 78., 54., 28.],
                     [50., 104., 162., 174., 186., 128., 66.],
                     [90., 186., 288., 306., 324., 222., 114.],
                     [120., 246., 378., 396., 414., 282., 144.],
                     [90., 184., 282., 294., 306., 208., 106.],
                     [50., 102., 156., 162., 168., 114., 58.]],

                    [[60., 123., 189., 198., 207., 141., 72.],
                     [135., 276., 423., 441., 459., 312., 159.],
                     [225., 459., 702., 729., 756., 513., 261.],
                     [270., 549., 837., 864., 891., 603., 306.],
                     [195., 396., 603., 621., 639., 432., 219.],
                     [105., 213., 324., 333., 342., 231., 117.]],

                    [[60., 122., 186., 192., 198., 134., 68.],
                     [130., 264., 402., 414., 426., 288., 146.],
                     [210., 426., 648., 666., 684., 462., 234.],
                     [240., 486., 738., 756., 774., 522., 264.],
                     [170., 344., 522., 534., 546., 368., 186.],
                     [90., 182., 276., 282., 288., 194., 98.]],

                    [[40., 81., 123., 126., 129., 87., 44.],
                     [85., 172., 261., 267., 273., 184., 93.],
                     [135., 273., 414., 423., 432., 291., 147.],
                     [150., 303., 459., 468., 477., 321., 162.],
                     [105., 212., 321., 327., 333., 224., 113.],
                     [55., 111., 168., 171., 174., 117., 59.]]]]]).astype(np.float32)

    expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_3d')

**convtranspose_attributes**

::

    x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                    [3., 4., 5.],
                    [6., 7., 8.]]]]).astype(np.float32)

    W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                    [1., 1., 1.],
                    [1., 1., 1.]],
                   [[1., 1., 1.],
                    [1., 1., 1.],
                    [1., 1., 1.]]]]).astype(np.float32)

    y = np.array([[[[0., 0., 1., 1., 3., 2., 2., 0.],  # (1, 2, 10, 8)
                    [0., 0., 1., 1., 3., 2., 2., 0.],
                    [0., 0., 1., 1., 3., 2., 2., 0.],
                    [3., 3., 7., 4., 9., 5., 5., 0.],
                    [3., 3., 7., 4., 9., 5., 5., 0.],
                    [3., 3., 7., 4., 9., 5., 5., 0.],
                    [6., 6., 13., 7., 15., 8., 8., 0.],
                    [6., 6., 13., 7., 15., 8., 8., 0.],
                    [6., 6., 13., 7., 15., 8., 8., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.]],

                   [[0., 0., 1., 1., 3., 2., 2., 0.],
                    [0., 0., 1., 1., 3., 2., 2., 0.],
                    [0., 0., 1., 1., 3., 2., 2., 0.],
                    [3., 3., 7., 4., 9., 5., 5., 0.],
                    [3., 3., 7., 4., 9., 5., 5., 0.],
                    [3., 3., 7., 4., 9., 5., 5., 0.],
                    [6., 6., 13., 7., 15., 8., 8., 0.],
                    [6., 6., 13., 7., 15., 8., 8., 0.],
                    [6., 6., 13., 7., 15., 8., 8., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0.]]]]).astype(np.float32)

    node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"],
                                 strides=[3, 2],
                                 output_shape=[10, 8])
    expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_output_shape')

    node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"],
                                 strides=[3, 2],
                                 output_padding=[1, 1])
    expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_pad')

    node = onnx.helper.make_node(
        'ConvTranspose', ['X', 'W'], ['Y'],
        name='test',
        strides=[3, 2],
        output_shape=[10, 8],
        kernel_shape=[3, 3],
        output_padding=[1, 1]
    )
    expect(node, inputs=[x, W], outputs=[y],
           name='test_convtranspose_kernel_shape')

**convtranspose_pads**

::

    x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                    [3., 4., 5.],
                    [6., 7., 8.]]]]).astype(np.float32)

    W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                    [1., 1., 1.],
                    [1., 1., 1.]],
                   [[1., 1., 1.],
                    [1., 1., 1.],
                    [1., 1., 1.]]]]).astype(np.float32)

    node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"],
                                 strides=[3, 2],
                                 pads=[1, 2, 1, 2])

    y = np.array([[[[1., 1., 3.],  # (1, 2, 7, 3)
                    [1., 1., 3.],
                    [7., 4., 9.],
                    [7., 4., 9.],
                    [7., 4., 9.],
                    [13., 7., 15.],
                    [13., 7., 15.]],

                   [[1., 1., 3.],
                    [1., 1., 3.],
                    [7., 4., 9.],
                    [7., 4., 9.],
                    [7., 4., 9.],
                    [13., 7., 15.],
                    [13., 7., 15.]]]]).astype(np.float32)

    expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_pads')

**convtranspose_dilations**

::

    x = np.array([[[[3., 8., 1.],  # (1, 1, 3, 3)
                    [9., 5., 7.],
                    [3., 2., 6.]]]]).astype(np.float32)
    W = np.array([[[[7., 2.],  # (1, 1, 2, 2)
                    [1., 9.]]]]).astype(np.float32)

    node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"], dilations=[2, 2])

    y = np.array([[[[21., 56., 13., 16., 2.],  # [1, 1, 5, 5]
                    [63., 35., 67., 10., 14.],
                    [24., 22., 76., 76., 21.],
                    [9., 5., 88., 45., 63.],
                    [3., 2., 33., 18., 54.]]]]).astype(np.float32)

    expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_dilations')

**convtranspose_autopad_same**

::

    x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                    [3., 4., 5.],
                    [6., 7., 8.]]]]).astype(np.float32)

    W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                    [1., 1., 1.],
                    [1., 1., 1.]],
                   [[1., 1., 1.],
                    [1., 1., 1.],
                    [1., 1., 1.]]]]).astype(np.float32)

    node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"], auto_pad="SAME_UPPER", strides=[2, 2])

    y = np.array([[[[0., 0., 1., 1., 3., 2.],
                    [0., 0., 1., 1., 3., 2.],
                    [3., 3., 8., 5., 12., 7.],
                    [3., 3., 7., 4., 9., 5.],
                    [9., 9., 20., 11., 24., 13.],
                    [6., 6., 13., 7., 15., 8.]],

                   [[0., 0., 1., 1., 3., 2.],
                    [0., 0., 1., 1., 3., 2.],
                    [3., 3., 8., 5., 12., 7.],
                    [3., 3., 7., 4., 9., 5.],
                    [9., 9., 20., 11., 24., 13.],
                    [6., 6., 13., 7., 15., 8.]]]]).astype(np.float32)

    expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_autopad_same')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The convolution transpose operator consumes an input tensor and a filter,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The convolution transpose operator consumes an input tensor and a filter,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">and computes the output.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">and computes the output.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If the pads parameter is provided the shape of the output is calculated via the following equation:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If the pads parameter is provided the shape of the output is calculated via the following equation:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">output_shape can also be explicitly specified in which case pads values are auto generated using these equations:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">output_shape can also be explicitly specified in which case pads values are auto generated using these equations:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]</code></td></tr>
    <tr style="1px solid black;"><td><code>10</code></td><td><code>10</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  If (auto_pads <span style="color:#BA4A00;">!</span>= SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  If (auto_pads =<span style="color:#196F3D;">=</span> SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **auto_pad**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **auto_pad**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Where default value is NOTSET, which means explicit padding is used.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Where default value is NOTSET, which means explicit padding is used.</code></td></tr>
    <tr style="1px solid black;"><td><code>18</code></td><td><code>18</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  SAME_UPPER or SAME_LOWER mean pad the input so that <span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span>output</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  SAME_UPPER or SAME_LOWER mean pad the input so that output<span style="color:#196F3D;">_</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">[</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">]</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">19</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  spatial size match the input.In case of odd number add the extra</code></td><td></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">19</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  = input_shape[i] * strides[i] for each axis i. The padding is</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">20</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  split between the two sides equally or almost equally (depending on</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">21</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  whether it is even or odd). In case the padding is an odd number,</code></td></tr>
    <tr style="1px solid black;"><td><code>20</code></td><td><code>22</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  padding at the end for SAME_UPPER and at the<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;"> </span>padding <span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span>a<span style="color:#196F3D;">d</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span>t the end for SAME_UPPER and at the</code></td></tr>
    <tr style="1px solid black;"><td><code>21</code></td><td><code>23</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  SAME_LOWER. <span style="color:#BA4A00;">V</span><span style="color:#BA4A00;">A</span><span style="color:#BA4A00;">L</span><span style="color:#BA4A00;">I</span>D<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">m</span>e<span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;">.</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">D</span><span style="color:#BA4A00;">e</span>fault value is 'NOTSET'.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span>SAME_LOWER. Default value is 'NOTSET'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **dilations**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **dilations**:</code></td></tr>
    <tr style="1px solid black;"><td><code>23</code></td><td><code>25</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  dilation value along each spatial axis of the filter.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  dilation value along each spatial axis of the filter.<span style="color:#196F3D;"> </span><span style="color:#196F3D;">I</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">t</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">26</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  present, the dilation defaults to 1 along each spatial axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **group**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **group**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  number of groups input channels and output channels are divided</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  number of groups input channels and output channels are divided</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  into. Default value is 1.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  into. Default value is 1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **kernel_shape**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **kernel_shape**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The shape of the convolution kernel. If not present, should be</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The shape of the convolution kernel. If not present, should be</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  inferred from input W.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  inferred from input W.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output_padding**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output_padding**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">34</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Additional elements added to the side with higher coordinate indices</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">35</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  in the output. Each padding value in "output_padding" must be less</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">36</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  than the corresponding stride/dilation dimension. By default, this</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">37</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  attribute is a zero vector. Note that this attribute doesn't</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">38</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  directly affect the computed output values. It only controls the</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">39</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  selection of the computed values, so changing this attribute only</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">40</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  adds or removes output elements. If "output_shape" is explicitly</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">41</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  provided, "output_padding" does not contribute additional size to</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">42</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  "output_shape" but participates in the computation of the needed</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">43</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  padding amount. This is also called adjs or adjustment in some</code></td></tr>
    <tr style="1px solid black;"><td><code>31</code></td><td><code>44</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  <span style="color:#BA4A00;">T</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">z</span><span style="color:#BA4A00;">e</span>r<span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">-</span><span style="color:#BA4A00;">p</span>a<span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">d</span>e<span style="color:#BA4A00;">d</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span>o<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span>s<span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">t</span>.<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">T</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">o</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">f</span>ra<span style="color:#196F3D;">m</span>e<span style="color:#196F3D;">w</span>o<span style="color:#196F3D;">r</span><span style="color:#196F3D;">k</span>s.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">32</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  called adjs/adjustment in some frameworks.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output_shape**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output_shape**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The shape of the output can be explicitly set which will cause pads</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The shape of the output can be explicitly set which will cause pads</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values to be auto generated. If output_shape is specified pads</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values to be auto generated. If output_shape is specified pads</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values are ignored. See doc for details for equations to generate</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  values are ignored. See doc for details for equations to generate</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  pads</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  pads</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **pads**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **pads**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Padding for the beginning and ending along each spatial axis, it can</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Padding for the beginning and ending along each spatial axis, it can</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  take any value greater than or equal to 0. The value represent the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  take any value greater than or equal to 0. The value represent the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  number of pixels added to the beginning and end part of the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  number of pixels added to the beginning and end part of the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding axis. pads format should be as follow [x1_begin,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding axis. pads format should be as follow [x1_begin,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the beginning of axis i and xi_end, the number of pixels</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the beginning of axis i and xi_end, the number of pixels</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the end of axis i. This attribute cannot be used</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the end of axis i. This attribute cannot be used</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  simultaneously with auto_pad attribute. If not present, the padding</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  simultaneously with auto_pad attribute. If not present, the padding</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  defaults to 0 along start and end of each spatial axis.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  defaults to 0 along start and end of each spatial axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **strides**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **strides**:</code></td></tr>
    <tr style="1px solid black;"><td><code>49</code></td><td><code>61</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Stride along each spatial axis.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Stride along each spatial axis.<span style="color:#196F3D;"> </span><span style="color:#196F3D;">I</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">s</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">62</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  to 1 along each spatial axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 3 inputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 3 inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input data tensor from previous layer; has size (N x C x H x W),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input data tensor from previous layer; has size (N x C x H x W),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  where N is the batch size, C is the number of channels, and H and W</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  where N is the batch size, C is the number of channels, and H and W</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  are the height and width. Note that this is for the 2D image.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  are the height and width. Note that this is for the 2D image.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Otherwise the size is (N x C x D1 x D2 ... x Dn)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Otherwise the size is (N x C x D1 x D2 ... x Dn)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **W** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **W** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The weight tensor that will be used in the convolutions; has size (C</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The weight tensor that will be used in the convolutions; has size (C</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x M/group x kH x kW), where C is the number of channels, and kH and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x M/group x kH x kW), where C is the number of channels, and kH and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  kW are the height and width of the kernel, and M is the number of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  kW are the height and width of the kernel, and M is the number of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  feature maps. For more than 2 dimensions, the weight shape will be</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  feature maps. For more than 2 dimensions, the weight shape will be</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (C x M/group x k1 x k2 x ... x kn), where (k1 x k2 x ... x kn) is</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (C x M/group x k1 x k2 x ... x kn), where (k1 x k2 x ... x kn) is</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the dimension of the kernel. The number of channels in the output</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the dimension of the kernel. The number of channels in the output</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  should be equal to W.shape[1] * group (assuming zero based indices</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  should be equal to W.shape[1] * group (assuming zero based indices</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  of the shape array)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  of the shape array)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional 1D bias to be added to the convolution, has size of M.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional 1D bias to be added to the convolution, has size of M.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output data tensor that contains the result of the convolution. The</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output data tensor that contains the result of the convolution. The</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output dimensions are functions of the kernel size, stride size, pad</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output dimensions are functions of the kernel size, stride size, pad</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  lengths and group count. The number of channels in the output should</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  lengths and group count. The number of channels in the output should</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  be equal to W.shape[1] * group (assuming zero based indices of the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  be equal to W.shape[1] * group (assuming zero based indices of the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  shape array)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  shape array)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    </table>

.. _l-onnx-op-convtranspose-1:

ConvTranspose - 1
=================

**Version**

* **name**: `ConvTranspose (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1**.

**Summary**

The convolution transpose operator consumes an input tensor and a filter,
and computes the output.

If the pads parameter is provided the shape of the output is calculated via the following equation:

  output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]

output_shape can also be explicitly specified in which case pads values are auto generated using these equations:

  total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
  If (auto_pads != SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
  Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).

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
* **output_padding**:
  The zero-padding added to one side of the output. This is also
  called adjs/adjustment in some frameworks.
* **output_shape**:
  The shape of the output can be explicitly set which will cause pads
  values to be auto generated. If output_shape is specified pads
  values are ignored. See doc for details for equations to generate
  pads
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
  Otherwise the size is (N x C x D1 x D2 ... x Dn)
* **W** (heterogeneous) - **T**:
  The weight tensor that will be used in the convolutions; has size (C
  x M/group x kH x kW), where C is the number of channels, and kH and
  kW are the height and width of the kernel, and M is the number of
  feature maps. For more than 2 dimensions, the weight shape will be
  (C x M/group x k1 x k2 x ... x kn), where (k1 x k2 x ... x kn) is
  the dimension of the kernel. The number of channels in the output
  should be equal to W.shape[1] * group (assuming zero based indices
  of the shape array)
* **B** (optional, heterogeneous) - **T**:
  Optional 1D bias to be added to the convolution, has size of M.

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output data tensor that contains the result of the convolution. The
  output dimensions are functions of the kernel size, stride size, pad
  lengths and group count. The number of channels in the output should
  be equal to W.shape[1] * group (assuming zero based indices of the
  shape array)

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
