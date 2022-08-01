
.. _l-onnx-doccom.ms.internal.nhwc-MaxPool:

==============================
com.ms.internal.nhwc - MaxPool
==============================

.. contents::
    :local:


.. _l-onnx-opcom-ms-internal-nhwc-maxpool-11:

MaxPool - 11 (com.ms.internal.nhwc)
===================================

**Version**

* **name**: `MaxPool (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.ms.internal.nhwc.MaxPool>`_
* **domain**: **com.ms.internal.nhwc**
* **since_version**: **11**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 11 of domain com.ms.internal.nhwc**.

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

* **activation**:
 Default value is ``?``.
* **activation_params**:
 Default value is ``?``.
* **auto_pad**:
  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID.
  Where default value is NOTSET, which means explicit padding is used.
  SAME_UPPER or SAME_LOWER mean pad the input so that the output
  spatial size match the input.In case of odd number add the extra
  padding at the end for SAME_UPPER and at the beginning for
  SAME_LOWER. VALID mean no padding. Default value is ``?``.
* **ceil_mode**:
  Whether to use ceil or floor (default) to compute the output shape. Default value is ``?``.
* **dilations**:
  Dilation value along each spatial axis of filter. If not present,
  the dilation defaults to 1 along each spatial axis. Default value is ``?``.
* **kernel_shape** (required):
  The size of the kernel along each axis. Default value is ``?``.
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
* **storage_order**:
  The storage order of the tensor. 0 is row major, and 1 is column
  major. Default value is ``?``.
* **strides**:
  Stride along each spatial axis. If not present, the stride defaults
  to 1 along each spatial axis. Default value is ``?``.

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
