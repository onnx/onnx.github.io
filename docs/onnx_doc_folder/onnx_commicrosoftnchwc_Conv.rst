
.. _l-onnx-doccom.microsoft.nchwc-Conv:

==========================
com.microsoft.nchwc - Conv
==========================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-nchwc-conv-1:

Conv - 1 (com.microsoft.nchwc)
==============================

**Version**

* **name**: `Conv (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.nchwc.Conv>`_
* **domain**: **com.microsoft.nchwc**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft.nchwc**.

**Summary**

For internal use.

**Attributes**

* **activation**:
 Default value is ``?``.
* **activation_params**:
 Default value is ``?``.
* **auto_pad**:
 Default value is ``?``.
* **dilations**:
 Default value is ``?``.
* **group**:
 Default value is ``?``.
* **kernel_shape**:
 Default value is ``?``.
* **pads**:
 Default value is ``?``.
* **strides**:
 Default value is ``?``.

**Inputs**

Between 2 and 4 inputs.

* **X** (heterogeneous) - **T**:

* **W** (heterogeneous) - **T**:

* **B** (optional, heterogeneous) - **T**:

* **Sum** (optional, heterogeneous) - **T**:

**Outputs**

* **Y** (heterogeneous) - **T**:

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
