
.. _l-onnx-doc-MaxPool:

=======
MaxPool
=======

.. contents::
    :local:


.. _l-onnx-op-maxpool-12:

MaxPool - 12
============

**Version**

* **name**: `MaxPool (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool>`_
* **domain**: **main**
* **since_version**: **12**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 12**.

**Summary**

MaxPool consumes an input tensor X and applies max pooling across
the tensor according to kernel sizes, stride sizes, and pad lengths.
max pooling consisting of computing the max on all values of a
subset of the input tensor according to the kernel size and downsampling the
data into the output tensor Y for further processing. The output spatial shape will be following:
::

    output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)

or
::

    output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)

if ceil_mode is enabled

::

    * pad_shape[i] is sum of pads along axis i

`auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
::

    VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
    SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])

And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
::

    pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]

The output of each pooling window is maximum number of elements exclude pad.

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
* **ceil_mode**:
  Whether to use ceil or floor (default) to compute the output shape. Default value is ``0``.
* **dilations**:
  Dilation value along each spatial axis of filter. If not present,
  the dilation defaults to 1 along each spatial axis.
* **kernel_shape** (required):
  The size of the kernel along each axis.
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
* **storage_order**:
  The storage order of the tensor. 0 is row major, and 1 is column
  major. Default value is ``0``.
* **strides**:
  Stride along each spatial axis. If not present, the stride defaults
  to 1 along each spatial axis.

**Inputs**

* **X** (heterogeneous) - **T**:
  Input data tensor from the previous operator; dimensions for image
  case are (N x C x H x W), where N is the batch size, C is the number
  of channels, and H and W are the height and the width of the data.
  For non image case, the dimensions are in the form of (N x C x D1 x
  D2 ... Dn), where N is the batch size. Optionally, if dimension
  denotation is in effect, the operation expects the input data tensor
  to arrive with the dimension denotation of [DATA_BATCH,
  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].

**Outputs**

Between 1 and 2 outputs.

* **Y** (heterogeneous) - **T**:
  Output data tensor from average or max pooling across the input
  tensor. Dimensions will vary based on various kernel, stride, and
  pad sizes. Floor value of the dimension is used
* **Indices** (optional, heterogeneous) - **I**:
  Indices tensor from max pooling across the input tensor. The
  dimensions of indices are the same as output tensor. The values in
  indices of are the indices of the selected values during pooling.
  The indices are computed as flatten 1-D tensor, and the indices do
  not consider padding. So the values in indices are in [0, N x C x D1
  x ... x Dn).

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int8),
  tensor(uint8)
  ):
  Constrain input and output types to float and 8 bit tensors.
* **I** in (
  tensor(int64)
  ):
  Constrain index tensor to int64

**Examples**

**maxpool_2d_uint8**

::

    """
    input_shape: [1, 1, 5, 5]
    output_shape: [1, 1, 5, 5]
    pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
    """
    node = onnx.helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[5, 5],
        pads=[2, 2, 2, 2]
    )
    x = np.array([[[
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ]]]).astype(np.uint8)
    y = np.array([[[
        [13, 14, 15, 15, 15],
        [18, 19, 20, 20, 20],
        [23, 24, 25, 25, 25],
        [23, 24, 25, 25, 25],
        [23, 24, 25, 25, 25]]]]).astype(np.uint8)

    expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_uint8')

**maxpool_2d_precomputed_pads**

::

    """
    input_shape: [1, 1, 5, 5]
    output_shape: [1, 1, 5, 5]
    pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
    """
    node = onnx.helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[5, 5],
        pads=[2, 2, 2, 2]

    )
    x = np.array([[[
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ]]]).astype(np.float32)
    y = np.array([[[
        [13, 14, 15, 15, 15],
        [18, 19, 20, 20, 20],
        [23, 24, 25, 25, 25],
        [23, 24, 25, 25, 25],
        [23, 24, 25, 25, 25]]]]).astype(np.float32)

    expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_precomputed_pads')

**maxpool_with_argmax_2d_precomputed_pads**

::

    """
    input_shape: [1, 1, 5, 5]
    output_shape: [1, 1, 5, 5]
    pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
    """
    node = onnx.helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y', 'z'],
        kernel_shape=[5, 5],
        pads=[2, 2, 2, 2]
    )
    x = np.array([[[
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ]]]).astype(np.float32)
    y = np.array([[[
        [13, 14, 15, 15, 15],
        [18, 19, 20, 20, 20],
        [23, 24, 25, 25, 25],
        [23, 24, 25, 25, 25],
        [23, 24, 25, 25, 25]]]]).astype(np.float32)
    z = np.array([[[
        [12, 13, 14, 14, 14],
        [17, 18, 19, 19, 19],
        [22, 23, 24, 24, 24],
        [22, 23, 24, 24, 24],
        [22, 23, 24, 24, 24]]]]).astype(np.int64)

    expect(node, inputs=[x], outputs=[y, z], name='test_maxpool_with_argmax_2d_precomputed_pads')

**maxpool_2d_precomputed_strides**

::

    """
    input_shape: [1, 1, 5, 5]
    output_shape: [1, 1, 2, 2]
    """
    node = onnx.helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[2, 2],
        strides=[2, 2]
    )
    x = np.array([[[
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ]]]).astype(np.float32)
    y = np.array([[[[7, 9],
                    [17, 19]]]]).astype(np.float32)

    expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_precomputed_strides')

**maxpool_with_argmax_2d_precomputed_strides**

::

    """
    input_shape: [1, 1, 5, 5]
    output_shape: [1, 1, 2, 2]
    """
    node = onnx.helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y', 'z'],
        kernel_shape=[2, 2],
        strides=[2, 2],
        storage_order=1
    )
    x = np.array([[[
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ]]]).astype(np.float32)
    y = np.array([[[[7, 9],
                    [17, 19]]]]).astype(np.float32)
    z = np.array([[[[6, 16],
                    [8, 18]]]]).astype(np.int64)

    expect(node, inputs=[x], outputs=[y, z], name='test_maxpool_with_argmax_2d_precomputed_strides')

**maxpool_2d_precomputed_same_upper**

::

    """
    input_shape: [1, 1, 5, 5]
    output_shape: [1, 1, 3, 3]
    pad_shape: [2, 2] -> [1, 1, 1, 1] by axis
    """
    node = onnx.helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[3, 3],
        strides=[2, 2],
        auto_pad='SAME_UPPER'
    )
    x = np.array([[[
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ]]]).astype(np.float32)
    y = np.array([[[[7, 9, 10],
                    [17, 19, 20],
                    [22, 24, 25]]]]).astype(np.float32)

    expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_precomputed_same_upper')

**maxpool_1d_default**

::

    """
    input_shape: [1, 3, 32]
    output_shape: [1, 3, 31]
    """
    node = onnx.helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[2],
    )
    x = np.random.randn(1, 3, 32).astype(np.float32)
    x_shape = np.shape(x)
    kernel_shape = [2]
    strides = [1]
    out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
    padded = x
    y = pool(padded, x_shape, kernel_shape, strides, out_shape, [0], 'MAX')

    expect(node, inputs=[x], outputs=[y], name='test_maxpool_1d_default')

**maxpool_2d_default**

::

    """
    input_shape: [1, 3, 32, 32]
    output_shape: [1, 3, 31, 31]
    """
    node = onnx.helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[2, 2],
    )
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x_shape = np.shape(x)
    kernel_shape = (2, 2)
    strides = (1, 1)
    out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
    padded = x
    y = pool(padded, x_shape, kernel_shape, strides, out_shape, (0, 0), 'MAX')

    expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_default')

**maxpool_3d_default**

::

    """
    input_shape: [1, 3, 32, 32, 32]
    output_shape: [1, 3, 31, 31, 31]
    """
    node = onnx.helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[2, 2, 2],
    )
    x = np.random.randn(1, 3, 32, 32, 32).astype(np.float32)
    x_shape = np.shape(x)
    kernel_shape = [2, 2, 2]
    strides = [1, 1, 1]
    out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
    padded = x
    y = pool(padded, x_shape, kernel_shape, strides, out_shape, [0, 0, 0], 'MAX')

    expect(node, inputs=[x], outputs=[y], name='test_maxpool_3d_default')

**maxpool_2d_same_upper**

::

    """
    input_shape: [1, 3, 32, 32]
    output_shape: [1, 3, 32, 32]
    pad_shape: [1, 1] -> [0, 1, 0, 1] by axis
    """
    node = onnx.helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[2, 2],
        auto_pad='SAME_UPPER'
    )
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x_shape = np.shape(x)
    kernel_shape = (2, 2)
    strides = (1, 1)
    out_shape = get_output_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides)
    pad_shape = get_pad_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides, out_shape)
    pad_top = pad_shape[0] // 2
    pad_bottom = pad_shape[0] - pad_top
    pad_left = pad_shape[1] // 2
    pad_right = pad_shape[1] - pad_left
    padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                    constant_values=np.nan)
    y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'MAX')

    expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_same_upper')

**maxpool_2d_same_lower**

::

    """
    input_shape: [1, 3, 32, 32]
    output_shape: [1, 3, 32, 32]
    pad_shape: [1, 1] -> [1, 0, 1, 0] by axis
    """
    node = onnx.helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[2, 2],
        auto_pad='SAME_LOWER'
    )
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x_shape = np.shape(x)
    kernel_shape = (2, 2)
    strides = (1, 1)
    out_shape = get_output_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides)
    pad_shape = get_pad_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides, out_shape)
    pad_bottom = pad_shape[0] // 2
    pad_top = pad_shape[0] - pad_bottom
    pad_right = pad_shape[1] // 2
    pad_left = pad_shape[1] - pad_right
    padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                    constant_values=np.nan)
    y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'MAX')

    expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_same_lower')

**maxpool_2d_pads**

::

    """
    input_shape: [1, 3, 28, 28]
    output_shape: [1, 3, 30, 30]
    pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
    """
    node = onnx.helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[3, 3],
        pads=[2, 2, 2, 2]
    )
    x = np.random.randn(1, 3, 28, 28).astype(np.float32)
    x_shape = np.shape(x)
    kernel_shape = (3, 3)
    strides = (1, 1)
    pad_bottom = pad_top = pad_right = pad_left = 2
    pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
    out_shape = get_output_shape('VALID', np.add(x_shape[2:], pad_shape), kernel_shape, strides)
    padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                    constant_values=np.nan)
    y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'MAX')

    expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_pads')

**maxpool_2d_strides**

::

    """
    input_shape: [1, 3, 32, 32]
    output_shape: [1, 3, 10, 10]
    """
    node = onnx.helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[5, 5],
        strides=[3, 3]
    )
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x_shape = np.shape(x)
    kernel_shape = (5, 5)
    strides = (3, 3)
    out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
    padded = x
    y = pool(padded, x_shape, kernel_shape, strides, out_shape, (0, 0), 'MAX')

    expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_strides')

**maxpool_2d_ceil**

::

    """
    input_shape: [1, 1, 4, 4]
    output_shape: [1, 1, 2, 2]
    """
    node = onnx.helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[3, 3],
        strides=[2, 2],
        ceil_mode=True
    )
    x = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]]).astype(np.float32)
    y = np.array([[[
        [11, 12],
        [15, 16]]]]).astype(np.float32)

    expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_ceil')

**maxpool_2d_dilations**

::

    """
    input_shape: [1, 1, 4, 4]
    output_shape: [1, 1, 2, 2]
    """
    node = onnx.helper.make_node(
        'MaxPool',
        inputs=['x'],
        outputs=['y'],
        kernel_shape=[2, 2],
        strides=[1, 1],
        dilations=[2, 2]
    )
    x = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]]).astype(np.float32)
    y = np.array([[[
        [11, 12],
        [15, 16]]]]).astype(np.float32)

    expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_dilations')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">MaxPool consumes an input tensor X and applies max pooling across</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">MaxPool consumes an input tensor X and applies max pooling across</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the tensor according to kernel sizes, stride sizes, and pad lengths.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the tensor according to kernel sizes, stride sizes, and pad lengths.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">max pooling consisting of computing the max on all values of a</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">max pooling consisting of computing the max on all values of a</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">subset of the input tensor according to the kernel size and downsampling the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">subset of the input tensor according to the kernel size and downsampling the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">data into the output tensor Y for further processing. The output spatial shape will be following:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">data into the output tensor Y for further processing. The output spatial shape will be following:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">or</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">or</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">if ceil_mode is enabled</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">if ceil_mode is enabled</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    * pad_shape[i] is sum of pads along axis i</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    * pad_shape[i] is sum of pads along axis i</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">auto_pad is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">auto_pad is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">And pad shape will be following if SAME_UPPER or SAME_LOWER:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">And pad shape will be following if SAME_UPPER or SAME_LOWER:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The output of each pooling window is maximum number of elements exclude pad.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The output of each pooling window is maximum number of elements exclude pad.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **auto_pad**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **auto_pad**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Where default value is NOTSET, which means explicit padding is used.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Where default value is NOTSET, which means explicit padding is used.</code></td></tr>
    <tr style="1px solid black;"><td><code>38</code></td><td><code>38</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  SAME_UPPER or SAME_LOWER mean pad the input so that <span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span>output</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  SAME_UPPER or SAME_LOWER mean pad the input so that output<span style="color:#196F3D;">_</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">[</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">]</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">39</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  spatial size match the input.In case of odd number add the extra</code></td><td></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">39</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  = ceil(input_shape[i] / strides[i]) for each axis i. The padding</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">40</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  is split between the two sides equally or almost equally (depending</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">41</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  on whether it is even or odd). In case the padding is an odd number,</code></td></tr>
    <tr style="1px solid black;"><td><code>40</code></td><td><code>42</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  padding at the end for SAME_UPPER and at the<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;"> </span>padding <span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span>a<span style="color:#196F3D;">d</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span>t the end for SAME_UPPER and at the</code></td></tr>
    <tr style="1px solid black;"><td><code>41</code></td><td><code>43</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  SAME_LOWER. <span style="color:#BA4A00;">V</span><span style="color:#BA4A00;">A</span><span style="color:#BA4A00;">L</span><span style="color:#BA4A00;">I</span>D<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">m</span>e<span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;">.</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">D</span><span style="color:#BA4A00;">e</span>fault value is 'NOTSET'.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span>SAME_LOWER. Default value is 'NOTSET'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **ceil_mode**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **ceil_mode**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether to use ceil or floor (default) to compute the output shape. Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether to use ceil or floor (default) to compute the output shape. Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **dilations**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **dilations**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Dilation value along each spatial axis of filter. If not present,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Dilation value along each spatial axis of filter. If not present,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the dilation defaults to 1 along each spatial axis.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the dilation defaults to 1 along each spatial axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **kernel_shape** (required):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **kernel_shape** (required):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The size of the kernel along each axis.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The size of the kernel along each axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **pads**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **pads**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Padding for the beginning and ending along each spatial axis, it can</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Padding for the beginning and ending along each spatial axis, it can</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  take any value greater than or equal to 0. The value represent the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  take any value greater than or equal to 0. The value represent the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  number of pixels added to the beginning and end part of the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  number of pixels added to the beginning and end part of the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding axis. pads format should be as follow [x1_begin,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding axis. pads format should be as follow [x1_begin,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the beginning of axis i and xi_end, the number of pixels</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the beginning of axis i and xi_end, the number of pixels</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the end of axis i. This attribute cannot be used</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the end of axis i. This attribute cannot be used</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  simultaneously with auto_pad attribute. If not present, the padding</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  simultaneously with auto_pad attribute. If not present, the padding</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  defaults to 0 along start and end of each spatial axis.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  defaults to 0 along start and end of each spatial axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **storage_order**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **storage_order**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The storage order of the tensor. 0 is row major, and 1 is column</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The storage order of the tensor. 0 is row major, and 1 is column</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  major. Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  major. Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **strides**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **strides**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Stride along each spatial axis. If not present, the stride defaults</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Stride along each spatial axis. If not present, the stride defaults</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  to 1 along each spatial axis.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  to 1 along each spatial axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input data tensor from the previous operator; dimensions for image</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input data tensor from the previous operator; dimensions for image</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  case are (N x C x H x W), where N is the batch size, C is the number</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  case are (N x C x H x W), where N is the batch size, C is the number</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  of channels, and H and W are the height and the width of the data.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  of channels, and H and W are the height and the width of the data.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  For non image case, the dimensions are in the form of (N x C x D1 x</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  For non image case, the dimensions are in the form of (N x C x D1 x</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  D2 ... Dn), where N is the batch size. Optionally, if dimension</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  D2 ... Dn), where N is the batch size. Optionally, if dimension</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  denotation is in effect, the operation expects the input data tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  denotation is in effect, the operation expects the input data tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  to arrive with the dimension denotation of [DATA_BATCH,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  to arrive with the dimension denotation of [DATA_BATCH,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2 outputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2 outputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output data tensor from average or max pooling across the input</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output data tensor from average or max pooling across the input</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor. Dimensions will vary based on various kernel, stride, and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor. Dimensions will vary based on various kernel, stride, and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  pad sizes. Floor value of the dimension is used</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  pad sizes. Floor value of the dimension is used</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Indices** (optional, heterogeneous) - **I**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Indices** (optional, heterogeneous) - **I**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Indices tensor from max pooling across the input tensor. The</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Indices tensor from max pooling across the input tensor. The</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimensions of indices are the same as output tensor. The values in</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimensions of indices are the same as output tensor. The values in</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices of are the indices of the selected values during pooling.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices of are the indices of the selected values during pooling.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The indices are computed as flatten 1-D tensor, and the indices do</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The indices are computed as flatten 1-D tensor, and the indices do</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  not consider padding. So the values in indices are in [0, N x C x D1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  not consider padding. So the values in indices are in [0, N x C x D1</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x ... x Dn).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x ... x Dn).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td><code>99</code></td><td><code>101</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  tensor(float16)</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  tensor(float16)<span style="color:#196F3D;">,</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">102</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">103</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td><code>101</code></td><td><code>105</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Constrain input and output types to float tensors.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Constrain input and output types to float <span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">8</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">i</span>t<span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span>ensors.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **I** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **I** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">109</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain index tensor to int64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain index tensor to int64</code></td></tr>
    </table>

.. _l-onnx-op-maxpool-11:

MaxPool - 11
============

**Version**

* **name**: `MaxPool (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

MaxPool consumes an input tensor X and applies max pooling across
the tensor according to kernel sizes, stride sizes, and pad lengths.
max pooling consisting of computing the max on all values of a
subset of the input tensor according to the kernel size and downsampling the
data into the output tensor Y for further processing. The output spatial shape will be following:
::

    output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)

or
::

    output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)

if ceil_mode is enabled

::

    * pad_shape[i] is sum of pads along axis i

`auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
::

    VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
    SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])

And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
::

    pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]

The output of each pooling window is maximum number of elements exclude pad.

**Attributes**

* **auto_pad**:
  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.
  Where default value is NOTSET, which means explicit padding is used.
  SAME_UPPER or SAME_LOWER mean pad the input so that the output
  spatial size match the input.In case of odd number add the extra
  padding at the end for SAME_UPPER and at the beginning for
  SAME_LOWER. VALID mean no padding. Default value is ``'NOTSET'``.
* **ceil_mode**:
  Whether to use ceil or floor (default) to compute the output shape. Default value is ``0``.
* **dilations**:
  Dilation value along each spatial axis of filter. If not present,
  the dilation defaults to 1 along each spatial axis.
* **kernel_shape** (required):
  The size of the kernel along each axis.
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
* **storage_order**:
  The storage order of the tensor. 0 is row major, and 1 is column
  major. Default value is ``0``.
* **strides**:
  Stride along each spatial axis. If not present, the stride defaults
  to 1 along each spatial axis.

**Inputs**

* **X** (heterogeneous) - **T**:
  Input data tensor from the previous operator; dimensions for image
  case are (N x C x H x W), where N is the batch size, C is the number
  of channels, and H and W are the height and the width of the data.
  For non image case, the dimensions are in the form of (N x C x D1 x
  D2 ... Dn), where N is the batch size. Optionally, if dimension
  denotation is in effect, the operation expects the input data tensor
  to arrive with the dimension denotation of [DATA_BATCH,
  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].

**Outputs**

Between 1 and 2 outputs.

* **Y** (heterogeneous) - **T**:
  Output data tensor from average or max pooling across the input
  tensor. Dimensions will vary based on various kernel, stride, and
  pad sizes. Floor value of the dimension is used
* **Indices** (optional, heterogeneous) - **I**:
  Indices tensor from max pooling across the input tensor. The
  dimensions of indices are the same as output tensor. The values in
  indices of are the indices of the selected values during pooling.
  The indices are computed as flatten 1-D tensor, and the indices do
  not consider padding. So the values in indices are in [0, N x C x D1
  x ... x Dn).

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
* **I** in (
  tensor(int64)
  ):
  Constrain index tensor to int64

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">MaxPool consumes an input tensor X and applies max pooling across</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">MaxPool consumes an input tensor X and applies max pooling across</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the tensor according to kernel sizes, stride sizes, and pad lengths.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the tensor according to kernel sizes, stride sizes, and pad lengths.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">max pooling consisting of computing the max on all values of a</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">max pooling consisting of computing the max on all values of a</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">subset of the input tensor according to the kernel size and downsampling the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">subset of the input tensor according to the kernel size and downsampling the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">data into the output tensor Y for further processing. The output spatial shape will be following:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">data into the output tensor Y for further processing. The output spatial shape will be following:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">or</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">or</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">if ceil_mode is enabled</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">if ceil_mode is enabled</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    * pad_shape[i] is sum of pads along axis i</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    * pad_shape[i] is sum of pads along axis i</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">auto_pad is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">auto_pad is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">And pad shape will be following if SAME_UPPER or SAME_LOWER:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">And pad shape will be following if SAME_UPPER or SAME_LOWER:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The output of each pooling window is maximum number of elements exclude pad.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The output of each pooling window is maximum number of elements exclude pad.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **auto_pad**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **auto_pad**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Where default value is NOTSET, which means explicit padding is used.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Where default value is NOTSET, which means explicit padding is used.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  SAME_UPPER or SAME_LOWER mean pad the input so that the output</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  SAME_UPPER or SAME_LOWER mean pad the input so that the output</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  spatial size match the input.In case of odd number add the extra</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  spatial size match the input.In case of odd number add the extra</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  padding at the end for SAME_UPPER and at the beginning for</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  padding at the end for SAME_UPPER and at the beginning for</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  SAME_LOWER. VALID mean no padding. Default value is 'NOTSET'.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  SAME_LOWER. VALID mean no padding. Default value is 'NOTSET'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **ceil_mode**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **ceil_mode**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether to use ceil or floor (default) to compute the output shape. Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether to use ceil or floor (default) to compute the output shape. Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **dilations**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **dilations**:</code></td></tr>
    <tr style="1px solid black;"><td><code>45</code></td><td><code>45</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Dilation value along each spatial axis of filter.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Dilation value along each spatial axis of filter.<span style="color:#196F3D;"> </span><span style="color:#196F3D;">I</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">,</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">46</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  the dilation defaults to 1 along each spatial axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **kernel_shape** (required):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **kernel_shape** (required):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The size of the kernel along each axis.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The size of the kernel along each axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **pads**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **pads**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Padding for the beginning and ending along each spatial axis, it can</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Padding for the beginning and ending along each spatial axis, it can</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  take any value greater than or equal to 0. The value represent the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  take any value greater than or equal to 0. The value represent the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  number of pixels added to the beginning and end part of the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  number of pixels added to the beginning and end part of the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding axis. pads format should be as follow [x1_begin,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding axis. pads format should be as follow [x1_begin,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the beginning of axis i and xi_end, the number of pixels</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the beginning of axis i and xi_end, the number of pixels</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the end of axis i. This attribute cannot be used</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the end of axis i. This attribute cannot be used</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  simultaneously with auto_pad attribute. If not present, the padding</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  simultaneously with auto_pad attribute. If not present, the padding</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  defaults to 0 along start and end of each spatial axis.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  defaults to 0 along start and end of each spatial axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **storage_order**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **storage_order**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The storage order of the tensor. 0 is row major, and 1 is column</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The storage order of the tensor. 0 is row major, and 1 is column</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  major. Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  major. Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **strides**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **strides**:</code></td></tr>
    <tr style="1px solid black;"><td><code>62</code></td><td><code>63</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Stride along each spatial axis.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Stride along each spatial axis.<span style="color:#196F3D;"> </span><span style="color:#196F3D;">I</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">s</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">64</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  to 1 along each spatial axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input data tensor from the previous operator; dimensions for image</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input data tensor from the previous operator; dimensions for image</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  case are (N x C x H x W), where N is the batch size, C is the number</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  case are (N x C x H x W), where N is the batch size, C is the number</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  of channels, and H and W are the height and the width of the data.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  of channels, and H and W are the height and the width of the data.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  For non image case, the dimensions are in the form of (N x C x D1 x</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  For non image case, the dimensions are in the form of (N x C x D1 x</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  D2 ... Dn), where N is the batch size. Optionally, if dimension</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  D2 ... Dn), where N is the batch size. Optionally, if dimension</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  denotation is in effect, the operation expects the input data tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  denotation is in effect, the operation expects the input data tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  to arrive with the dimension denotation of [DATA_BATCH,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  to arrive with the dimension denotation of [DATA_BATCH,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2 outputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2 outputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output data tensor from average or max pooling across the input</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output data tensor from average or max pooling across the input</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor. Dimensions will vary based on various kernel, stride, and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor. Dimensions will vary based on various kernel, stride, and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  pad sizes. Floor value of the dimension is used</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  pad sizes. Floor value of the dimension is used</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Indices** (optional, heterogeneous) - **I**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Indices** (optional, heterogeneous) - **I**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Indices tensor from max pooling across the input tensor. The</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Indices tensor from max pooling across the input tensor. The</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimensions of indices are the same as output tensor. The values in</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimensions of indices are the same as output tensor. The values in</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices of are the indices of the selected values during pooling.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices of are the indices of the selected values during pooling.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The indices are computed as flatten 1-D tensor, and the indices do</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The indices are computed as flatten 1-D tensor, and the indices do</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  not consider padding. So the values in indices are in [0, N x C x D1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  not consider padding. So the values in indices are in [0, N x C x D1</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x ... x Dn).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x ... x Dn).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **I** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **I** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain index tensor to int64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain index tensor to int64</code></td></tr>
    </table>

.. _l-onnx-op-maxpool-10:

MaxPool - 10
============

**Version**

* **name**: `MaxPool (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool>`_
* **domain**: **main**
* **since_version**: **10**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 10**.

**Summary**

MaxPool consumes an input tensor X and applies max pooling across
the tensor according to kernel sizes, stride sizes, and pad lengths.
max pooling consisting of computing the max on all values of a
subset of the input tensor according to the kernel size and downsampling the
data into the output tensor Y for further processing. The output spatial shape will be following:
::

    output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)

or
::

    output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)

if ceil_mode is enabled

::

    * pad_shape[i] is sum of pads along axis i

`auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
::

    VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
    SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])

And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
::

    pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]

The output of each pooling window is maximum number of elements exclude pad.

**Attributes**

* **auto_pad**:
  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.
  Where default value is NOTSET, which means explicit padding is used.
  SAME_UPPER or SAME_LOWER mean pad the input so that the output
  spatial size match the input.In case of odd number add the extra
  padding at the end for SAME_UPPER and at the beginning for
  SAME_LOWER. VALID mean no padding. Default value is ``'NOTSET'``.
* **ceil_mode**:
  Whether to use ceil or floor (default) to compute the output shape. Default value is ``0``.
* **dilations**:
  Dilation value along each spatial axis of filter.
* **kernel_shape** (required):
  The size of the kernel along each axis.
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
* **storage_order**:
  The storage order of the tensor. 0 is row major, and 1 is column
  major. Default value is ``0``.
* **strides**:
  Stride along each spatial axis.

**Inputs**

* **X** (heterogeneous) - **T**:
  Input data tensor from the previous operator; dimensions for image
  case are (N x C x H x W), where N is the batch size, C is the number
  of channels, and H and W are the height and the width of the data.
  For non image case, the dimensions are in the form of (N x C x D1 x
  D2 ... Dn), where N is the batch size. Optionally, if dimension
  denotation is in effect, the operation expects the input data tensor
  to arrive with the dimension denotation of [DATA_BATCH,
  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].

**Outputs**

Between 1 and 2 outputs.

* **Y** (heterogeneous) - **T**:
  Output data tensor from average or max pooling across the input
  tensor. Dimensions will vary based on various kernel, stride, and
  pad sizes. Floor value of the dimension is used
* **Indices** (optional, heterogeneous) - **I**:
  Indices tensor from max pooling across the input tensor. The
  dimensions of indices are the same as output tensor. The values in
  indices of are the indices of the selected values during pooling.
  The indices are computed as flatten 1-D tensor, and the indices do
  not consider padding. So the values in indices are in [0, N x C x D1
  x ... x Dn).

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
* **I** in (
  tensor(int64)
  ):
  Constrain index tensor to int64

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">MaxPool consumes an input tensor X and applies max pooling across</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">MaxPool consumes an input tensor X and applies max pooling across</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the tensor according to kernel sizes, stride sizes, and pad lengths.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the tensor according to kernel sizes, stride sizes, and pad lengths.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">max pooling consisting of computing the max on all values of a</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">max pooling consisting of computing the max on all values of a</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">subset of the input tensor according to the kernel size and downsampling the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">subset of the input tensor according to the kernel size and downsampling the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">data into the output tensor Y for further processing. The output spatial shape will be following:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">data into the output tensor Y for further processing. The output spatial shape will be following:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td><code>7</code></td><td><code>7</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">    output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>    output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - <span style="color:#196F3D;">(</span><span style="color:#196F3D;">(</span>kernel_spatial_shape[i]<span style="color:#196F3D;"> </span><span style="color:#196F3D;">-</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">1</span>) <span style="color:#196F3D;">*</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">[</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">]</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">+</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;"> </span>/ strides_spatial_shape[i] + 1)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">9</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">or</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">10</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">::</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">11</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">12</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">    output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">13</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">14</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">if ceil_mode is enabled</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">15</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">16</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">::</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">17</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    * pad_shape[i] is sum of pads along axis i</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    * pad_shape[i] is sum of pads along axis i</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">auto_pad is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">auto_pad is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td><code>14</code></td><td><code>23</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">    VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>    VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - <span style="color:#196F3D;">(</span><span style="color:#196F3D;">(</span>kernel_spatial_shape[i] <span style="color:#196F3D;">-</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">*</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">[</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">]</span><span style="color:#196F3D;"> </span>+ 1) <span style="color:#196F3D;">+</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;"> </span>/ strides_spatial_shape[i])</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">And pad shape will be following if SAME_UPPER or SAME_LOWER:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">And pad shape will be following if SAME_UPPER or SAME_LOWER:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td><code>20</code></td><td><code>29</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">    pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>    pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + <span style="color:#196F3D;">(</span><span style="color:#196F3D;">(</span>kernel_spatial_shape[i] - <span style="color:#196F3D;">1</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">*</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">d</span>i<span style="color:#196F3D;">l</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span>n<span style="color:#196F3D;">s</span><span style="color:#196F3D;">[</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">]</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">+</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">-</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span>put_spatial_shape[i]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The output of each pooling window is maximum number of elements exclude pad.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The output of each pooling window is maximum number of elements exclude pad.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **auto_pad**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **auto_pad**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Where default value is NOTSET, which means explicit padding is used.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Where default value is NOTSET, which means explicit padding is used.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  SAME_UPPER or SAME_LOWER mean pad the input so that the output</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  SAME_UPPER or SAME_LOWER mean pad the input so that the output</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  spatial size match the input.In case of odd number add the extra</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  spatial size match the input.In case of odd number add the extra</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  padding at the end for SAME_UPPER and at the beginning for</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  padding at the end for SAME_UPPER and at the beginning for</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  SAME_LOWER. VALID mean no padding. Default value is 'NOTSET'.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  SAME_LOWER. VALID mean no padding. Default value is 'NOTSET'.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">42</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **ceil_mode**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">43</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Whether to use ceil or floor (default) to compute the output shape. Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">44</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **dilations**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">45</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Dilation value along each spatial axis of filter.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **kernel_shape** (required):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **kernel_shape** (required):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The size of the kernel along each axis.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The size of the kernel along each axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **pads**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **pads**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Padding for the beginning and ending along each spatial axis, it can</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Padding for the beginning and ending along each spatial axis, it can</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  take any value greater than or equal to 0. The value represent the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  take any value greater than or equal to 0. The value represent the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  number of pixels added to the beginning and end part of the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  number of pixels added to the beginning and end part of the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding axis. pads format should be as follow [x1_begin,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding axis. pads format should be as follow [x1_begin,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the beginning of axis i and xi_end, the number of pixels</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the beginning of axis i and xi_end, the number of pixels</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the end of axis i. This attribute cannot be used</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the end of axis i. This attribute cannot be used</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  simultaneously with auto_pad attribute. If not present, the padding</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  simultaneously with auto_pad attribute. If not present, the padding</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  defaults to 0 along start and end of each spatial axis.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  defaults to 0 along start and end of each spatial axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **storage_order**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **storage_order**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The storage order of the tensor. 0 is row major, and 1 is column</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The storage order of the tensor. 0 is row major, and 1 is column</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  major. Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  major. Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **strides**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **strides**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Stride along each spatial axis.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Stride along each spatial axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input data tensor from the previous operator; dimensions for image</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input data tensor from the previous operator; dimensions for image</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  case are (N x C x H x W), where N is the batch size, C is the number</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  case are (N x C x H x W), where N is the batch size, C is the number</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  of channels, and H and W are the height and the width of the data.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  of channels, and H and W are the height and the width of the data.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  For non image case, the dimensions are in the form of (N x C x D1 x</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  For non image case, the dimensions are in the form of (N x C x D1 x</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  D2 ... Dn), where N is the batch size. Optionally, if dimension</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  D2 ... Dn), where N is the batch size. Optionally, if dimension</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  denotation is in effect, the operation expects the input data tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  denotation is in effect, the operation expects the input data tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  to arrive with the dimension denotation of [DATA_BATCH,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  to arrive with the dimension denotation of [DATA_BATCH,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2 outputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2 outputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output data tensor from average or max pooling across the input</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output data tensor from average or max pooling across the input</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor. Dimensions will vary based on various kernel, stride, and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor. Dimensions will vary based on various kernel, stride, and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  pad sizes. Floor value of the dimension is used</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  pad sizes. Floor value of the dimension is used</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Indices** (optional, heterogeneous) - **I**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Indices** (optional, heterogeneous) - **I**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Indices tensor from max pooling across the input tensor. The</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Indices tensor from max pooling across the input tensor. The</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimensions of indices are the same as output tensor. The values in</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimensions of indices are the same as output tensor. The values in</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices of are the indices of the selected values during pooling.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  indices of are the indices of the selected values during pooling.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The indices are computed as flatten 1-D tensor, and the indices do</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The indices are computed as flatten 1-D tensor, and the indices do</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  not consider padding. So the values in indices are in [0, N x C x D1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  not consider padding. So the values in indices are in [0, N x C x D1</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x ... x Dn).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x ... x Dn).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **I** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **I** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain index tensor to int64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain index tensor to int64</code></td></tr>
    </table>

.. _l-onnx-op-maxpool-8:

MaxPool - 8
===========

**Version**

* **name**: `MaxPool (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool>`_
* **domain**: **main**
* **since_version**: **8**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 8**.

**Summary**

MaxPool consumes an input tensor X and applies max pooling across
the tensor according to kernel sizes, stride sizes, and pad lengths.
max pooling consisting of computing the max on all values of a
subset of the input tensor according to the kernel size and downsampling the
data into the output tensor Y for further processing. The output spatial shape will be following:
::

    output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)

    * pad_shape[i] is sum of pads along axis i

`auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
::

    VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
    SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])

And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
::

    pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]

The output of each pooling window is maximum number of elements exclude pad.

**Attributes**

* **auto_pad**:
  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.
  Where default value is NOTSET, which means explicit padding is used.
  SAME_UPPER or SAME_LOWER mean pad the input so that the output
  spatial size match the input.In case of odd number add the extra
  padding at the end for SAME_UPPER and at the beginning for
  SAME_LOWER. VALID mean no padding. Default value is ``'NOTSET'``.
* **kernel_shape** (required):
  The size of the kernel along each axis.
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
* **storage_order**:
  The storage order of the tensor. 0 is row major, and 1 is column
  major. Default value is ``0``.
* **strides**:
  Stride along each spatial axis.

**Inputs**

* **X** (heterogeneous) - **T**:
  Input data tensor from the previous operator; dimensions for image
  case are (N x C x H x W), where N is the batch size, C is the number
  of channels, and H and W are the height and the width of the data.
  For non image case, the dimensions are in the form of (N x C x D1 x
  D2 ... Dn), where N is the batch size. Optionally, if dimension
  denotation is in effect, the operation expects the input data tensor
  to arrive with the dimension denotation of [DATA_BATCH,
  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].

**Outputs**

Between 1 and 2 outputs.

* **Y** (heterogeneous) - **T**:
  Output data tensor from average or max pooling across the input
  tensor. Dimensions will vary based on various kernel, stride, and
  pad sizes. Floor value of the dimension is used
* **Indices** (optional, heterogeneous) - **I**:
  Indices tensor from max pooling across the input tensor. The
  dimensions of indices are the same as output tensor. The values in
  indices of are the indices of the selected values during pooling.
  The indices are computed as flatten 1-D tensor, and the indices do
  not consider padding. So the values in indices are in [0, N x C x D1
  x ... x Dn).

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
* **I** in (
  tensor(int64)
  ):
  Constrain index tensor to int64

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">MaxPool consumes an input tensor X and applies max pooling across</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">MaxPool consumes an input tensor X and applies max pooling across</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the tensor according to kernel sizes, stride sizes, and pad lengths.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the tensor according to kernel sizes, stride sizes, and pad lengths.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">max pooling consisting of computing the max on all values of a</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">max pooling consisting of computing the max on all values of a</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">subset of the input tensor according to the kernel size and downsampling the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">subset of the input tensor according to the kernel size and downsampling the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">data into the output tensor Y for further processing. The output spatial shape will be following:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">data into the output tensor Y for further processing. The output spatial shape will be following:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    * pad_shape[i] is sum of pads along axis i</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    * pad_shape[i] is sum of pads along axis i</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">auto_pad is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">auto_pad is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">And pad shape will be following if SAME_UPPER or SAME_LOWER:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">And pad shape will be following if SAME_UPPER or SAME_LOWER:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The output of each pooling window is maximum number of elements exclude pad.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The output of each pooling window is maximum number of elements exclude pad.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **auto_pad**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **auto_pad**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Where default value is NOTSET, which means explicit padding is used.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Where default value is NOTSET, which means explicit padding is used.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  SAME_UPPER or SAME_LOWER mean pad the input so that the output</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  SAME_UPPER or SAME_LOWER mean pad the input so that the output</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  spatial size match the input.In case of odd number add the extra</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  spatial size match the input.In case of odd number add the extra</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  padding at the end for SAME_UPPER and at the beginning for</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  padding at the end for SAME_UPPER and at the beginning for</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  SAME_LOWER. VALID mean no padding. Default value is 'NOTSET'.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  SAME_LOWER. VALID mean no padding. Default value is 'NOTSET'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **kernel_shape** (required):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **kernel_shape** (required):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The size of the kernel along each axis.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The size of the kernel along each axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **pads**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **pads**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Padding for the beginning and ending along each spatial axis, it can</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Padding for the beginning and ending along each spatial axis, it can</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  take any value greater than or equal to 0. The value represent the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  take any value greater than or equal to 0. The value represent the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  number of pixels added to the beginning and end part of the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  number of pixels added to the beginning and end part of the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding axis. pads format should be as follow [x1_begin,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  corresponding axis. pads format should be as follow [x1_begin,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the beginning of axis i and xi_end, the number of pixels</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the beginning of axis i and xi_end, the number of pixels</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the end of axis i. This attribute cannot be used</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  added at the end of axis i. This attribute cannot be used</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  simultaneously with auto_pad attribute. If not present, the padding</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  simultaneously with auto_pad attribute. If not present, the padding</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  defaults to 0 along start and end of each spatial axis.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  defaults to 0 along start and end of each spatial axis.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">45</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **storage_order**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">46</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  The storage order of the tensor. 0 is row major, and 1 is column</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">47</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  major. Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **strides**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **strides**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Stride along each spatial axis.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Stride along each spatial axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input data tensor from the previous operator; dimensions for image</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input data tensor from the previous operator; dimensions for image</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  case are (N x C x H x W), where N is the batch size, C is the number</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  case are (N x C x H x W), where N is the batch size, C is the number</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  of channels, and H and W are the height and the width of the data.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  of channels, and H and W are the height and the width of the data.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  For non image case, the dimensions are in the form of (N x C x D1 x</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  For non image case, the dimensions are in the form of (N x C x D1 x</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  D2 ... Dn), where N is the batch size. Optionally, if dimension</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  D2 ... Dn), where N is the batch size. Optionally, if dimension</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  denotation is in effect, the operation expects the input data tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  denotation is in effect, the operation expects the input data tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  to arrive with the dimension denotation of [DATA_BATCH,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  to arrive with the dimension denotation of [DATA_BATCH,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">65</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">Between 1 and 2 outputs.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">66</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output data tensor from average or max pooling across the input</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output data tensor from average or max pooling across the input</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor. Dimensions will vary based on various kernel, stride, and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor. Dimensions will vary based on various kernel, stride, and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  pad sizes. Floor value of the dimension is used</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  pad sizes. Floor value of the dimension is used</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">71</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **Indices** (optional, heterogeneous) - **I**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">72</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Indices tensor from max pooling across the input tensor. The</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">73</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  dimensions of indices are the same as output tensor. The values in</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">74</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  indices of are the indices of the selected values during pooling.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">75</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  The indices are computed as flatten 1-D tensor, and the indices do</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">76</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  not consider padding. So the values in indices are in [0, N x C x D1</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">77</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  x ... x Dn).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">87</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **I** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">88</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">89</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  ):</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">90</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Constrain index tensor to int64</code></td></tr>
    </table>

.. _l-onnx-op-maxpool-1:

MaxPool - 1
===========

**Version**

* **name**: `MaxPool (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1**.

**Summary**

MaxPool consumes an input tensor X and applies max pooling across
the tensor according to kernel sizes, stride sizes, and pad lengths.
max pooling consisting of computing the max on all values of a
subset of the input tensor according to the kernel size and downsampling the
data into the output tensor Y for further processing. The output spatial shape will be following:
::

    output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)

    * pad_shape[i] is sum of pads along axis i

`auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
::

    VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
    SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])

And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
::

    pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]

The output of each pooling window is maximum number of elements exclude pad.

**Attributes**

* **auto_pad**:
  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.
  Where default value is NOTSET, which means explicit padding is used.
  SAME_UPPER or SAME_LOWER mean pad the input so that the output
  spatial size match the input.In case of odd number add the extra
  padding at the end for SAME_UPPER and at the beginning for
  SAME_LOWER. VALID mean no padding. Default value is ``'NOTSET'``.
* **kernel_shape** (required):
  The size of the kernel along each axis.
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

* **X** (heterogeneous) - **T**:
  Input data tensor from the previous operator; dimensions for image
  case are (N x C x H x W), where N is the batch size, C is the number
  of channels, and H and W are the height and the width of the data.
  For non image case, the dimensions are in the form of (N x C x D1 x
  D2 ... Dn), where N is the batch size. Optionally, if dimension
  denotation is in effect, the operation expects the input data tensor
  to arrive with the dimension denotation of [DATA_BATCH,
  DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output data tensor from average or max pooling across the input
  tensor. Dimensions will vary based on various kernel, stride, and
  pad sizes. Floor value of the dimension is used

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
