
.. _l-onnx-doc-Resize:

======
Resize
======

.. contents::
    :local:


.. _l-onnx-op-resize-13:

Resize - 13
===========

**Version**

* **name**: `Resize (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize>`_
* **domain**: **main**
* **since_version**: **13**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 13**.

**Summary**

Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * (roi_end - roi_start) * scale) if input \"sizes\" is not specified.

**Attributes**

* **coordinate_transformation_mode**:
   This attribute describes how to transform the coordinate in the
  resized tensor to the coordinate in the original tensor. <br/>  The
  coordinate of each dimension is transformed individually. Let's
  describe a case using axis x as an example. Denote x_resized as the
  coordinate of axis x in the resized tensor, x_original as the
  coordinate of axis x in the original tensor, length_original as the
  length of the original tensor in axis x, length_resized as the
  length of the resized tensor in axis x, roi_x = (start_x, end_x) of
  the axis x in input "roi", scale = length_resized / length_original,
  <br/>  if coordinate_transformation_mode is "half_pixel", <br/>
  x_original = (x_resized + 0.5) / scale - 0.5, <br/>  if
  coordinate_transformation_mode is "pytorch_half_pixel", <br/>
  x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 :
  0, <br/>  if coordinate_transformation_mode is "align_corners",
  <br/> x_original = x_resized * (length_original - 1) /
  (length_resized - 1), <br/>  if coordinate_transformation_mode is
  "asymmetric", <br/> x_original = x_resized / scale, <br/>  if
  coordinate_transformation_mode is "tf_crop_and_resize", <br/>
  x_original = length_resized > 1 ? start_x * (length_original - 1) +
  x_resized * (end_x - start_x) * (length_original - 1) /
  (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original -
  1). Default value is ``'half_pixel'``.
* **cubic_coeff_a**:
  The coefficient 'a' used in cubic interpolation. Two common choice
  are -0.5 (in some cases of TensorFlow) and -0.75 (in PyTorch). Check
  out Equation (4) in https://ieeexplore.ieee.org/document/1163711 for
  the details. This attribute is valid only if "mode" is "cubic". Default value is ``-0.75``.
* **exclude_outside**:
  If set to 1, the weight of sampling locations outside the tensor
  will be set to 0 and the weight will be renormalized so that their
  sum is 1.0. The default value is 0. Default value is ``0``.
* **extrapolation_value**:
  When coordinate_transformation_mode is "tf_crop_and_resize" and
  x_original is outside the range [0, length_original - 1], this value
  is used as the corresponding output value. Default is 0.0f. Default value is ``0.0``.
* **mode**:
  Three interpolation modes: nearest (default), linear and cubic. The
  "linear" mode includes linear interpolation for 1D tensor and
  N-linear interpolation for N-D tensor (for example, bilinear
  interpolation for 2D tensor). The "cubic" mode includes cubic
  interpolation for 1D tensor and N-cubic interpolation for N-D tensor
  (for example, bicubic interpolation for 2D tensor). Default value is ``'nearest'``.
* **nearest_mode**:
  Four modes: round_prefer_floor (default, as known as round half
  down), round_prefer_ceil (as known as round half up), floor, ceil.
  Only used by nearest interpolation. It indicates how to get
  "nearest" pixel in input tensor from x_original, so this attribute
  is valid only if "mode" is "nearest". Default value is ``'round_prefer_floor'``.

**Inputs**

Between 1 and 4 inputs.

* **X** (heterogeneous) - **T1**:
  N-D tensor
* **roi** (optional, heterogeneous) - **T2**:
  1-D tensor given as [start1, ..., startN, end1, ..., endN], where N
  is the rank of X. The RoIs' coordinates are normalized in the
  coordinate system of the input image. It only takes effect when
  coordinate_transformation_mode is "tf_crop_and_resize"
* **scales** (optional, heterogeneous) - **tensor(float)**:
  The scale array along each dimension. It takes value greater than 0.
  If it's less than 1, it's sampling down, otherwise, it's upsampling.
  The number of elements of 'scales' should be the same as the rank of
  input 'X'. One of 'scales' and 'sizes' MUST be specified and it is
  an error if both are specified. If 'sizes' is needed, the user can
  use an empty string as the name of 'scales' in this operator's input
  list.
* **sizes** (optional, heterogeneous) - **tensor(int64)**:
  The size of the output tensor. The number of elements of 'sizes'
  should be the same as the rank of input 'X'. Only one of 'scales'
  and 'sizes' can be specified.

**Outputs**

* **Y** (heterogeneous) - **T1**:
  N-D tensor after resizing

**Type Constraints**

* **T1** in (
  tensor(bfloat16),
  tensor(bool),
  tensor(complex128),
  tensor(complex64),
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int16),
  tensor(int32),
  tensor(int64),
  tensor(int8),
  tensor(string),
  tensor(uint16),
  tensor(uint32),
  tensor(uint64),
  tensor(uint8)
  ):
  Constrain input 'X' and output 'Y' to all tensor types.
* **T2** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain roi type to float or double.

**Examples**

**resize_upsample_scales_nearest**

::

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', 'scales'],
        outputs=['Y'],
        mode='nearest',
    )

    data = np.array([[[
        [1, 2],
        [3, 4],
    ]]], dtype=np.float32)

    scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)

    # [[[[1. 1. 1. 2. 2. 2.]
    #    [1. 1. 1. 2. 2. 2.]
    #    [3. 3. 3. 4. 4. 4.]
    #    [3. 3. 3. 4. 4. 4.]]]]
    output = interpolate_nd(
        data, nearest_coeffs, scale_factors=scales).astype(np.float32)

    expect(node, inputs=[data, scales], outputs=[output],
           name='test_resize_upsample_scales_nearest')

**resize_downsample_scales_nearest**

::

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', 'scales'],
        outputs=['Y'],
        mode='nearest',
    )

    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]]], dtype=np.float32)

    scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

    # [[[[1. 3.]]]]
    output = interpolate_nd(
        data, nearest_coeffs, scale_factors=scales).astype(np.float32)

    expect(node, inputs=[data, scales], outputs=[output],
           name='test_resize_downsample_scales_nearest')

**resize_upsample_sizes_nearest**

::

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', '', 'sizes'],
        outputs=['Y'],
        mode='nearest',
    )

    data = np.array([[[
        [1, 2],
        [3, 4],
    ]]], dtype=np.float32)

    sizes = np.array([1, 1, 7, 8], dtype=np.int64)

    # [[[[1. 1. 1. 1. 2. 2. 2. 2.]
    #    [1. 1. 1. 1. 2. 2. 2. 2.]
    #    [1. 1. 1. 1. 2. 2. 2. 2.]
    #    [1. 1. 1. 1. 2. 2. 2. 2.]
    #    [3. 3. 3. 3. 4. 4. 4. 4.]
    #    [3. 3. 3. 3. 4. 4. 4. 4.]
    #    [3. 3. 3. 3. 4. 4. 4. 4.]]]]
    output = interpolate_nd(
        data, nearest_coeffs, output_size=sizes).astype(np.float32)

    expect(node, inputs=[data, sizes], outputs=[output],
           name='test_resize_upsample_sizes_nearest')

**resize_downsample_sizes_nearest**

::

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', '', 'sizes'],
        outputs=['Y'],
        mode='nearest',
    )

    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]]], dtype=np.float32)

    sizes = np.array([1, 1, 1, 3], dtype=np.int64)

    # [[[[1. 3.]]]]
    output = interpolate_nd(
        data, nearest_coeffs, output_size=sizes).astype(np.float32)

    expect(node, inputs=[data, sizes], outputs=[output],
           name='test_resize_downsample_sizes_nearest')

**resize_upsample_scales_linear**

::

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', 'scales'],
        outputs=['Y'],
        mode='linear',
    )

    data = np.array([[[
        [1, 2],
        [3, 4],
    ]]], dtype=np.float32)

    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    # [[[[1.   1.25 1.75 2.  ]
    #    [1.5  1.75 2.25 2.5 ]
    #    [2.5  2.75 3.25 3.5 ]
    #    [3.   3.25 3.75 4.  ]]]]
    output = interpolate_nd(
        data, linear_coeffs, scale_factors=scales).astype(np.float32)

    expect(node, inputs=[data, scales], outputs=[output],
           name='test_resize_upsample_scales_linear')

**resize_upsample_scales_linear_align_corners**

::

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', 'scales'],
        outputs=['Y'],
        mode='linear',
        coordinate_transformation_mode='align_corners'
    )

    data = np.array([[[
        [1, 2],
        [3, 4],
    ]]], dtype=np.float32)

    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    # [[[[1.         1.33333333 1.66666667 2.        ]
    #    [1.66666667 2.         2.33333333 2.66666667]
    #    [2.33333333 2.66666667 3.         3.33333333]
    #    [3.         3.33333333 3.66666667 4.        ]]]]
    output = interpolate_nd(
        data, linear_coeffs, scale_factors=scales, coordinate_transformation_mode='align_corners').astype(np.float32)

    expect(node, inputs=[data, scales], outputs=[output],
           name='test_resize_upsample_scales_linear_align_corners')

**resize_downsample_scales_linear**

::

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', 'scales'],
        outputs=['Y'],
        mode='linear',
    )

    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]]], dtype=np.float32)

    scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

    # [[[[2.6666665 4.3333331]]]]
    output = interpolate_nd(
        data, linear_coeffs, scale_factors=scales).astype(np.float32)

    expect(node, inputs=[data, scales], outputs=[output],
           name='test_resize_downsample_scales_linear')

**resize_downsample_scales_linear_align_corners**

::

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', 'scales'],
        outputs=['Y'],
        mode='linear',
        coordinate_transformation_mode='align_corners'
    )

    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]]], dtype=np.float32)

    scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

    # [[[[1.       3.142857]]]]
    output = interpolate_nd(
        data, linear_coeffs, scale_factors=scales, coordinate_transformation_mode='align_corners').astype(np.float32)

    expect(node, inputs=[data, scales], outputs=[output],
           name='test_resize_downsample_scales_linear_align_corners')

**resize_upsample_scales_cubic**

::

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', 'scales'],
        outputs=['Y'],
        mode='cubic',
    )

    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=np.float32)

    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    # [[[[ 0.47265625  0.76953125  1.24609375  1.875       2.28125
    #      2.91015625  3.38671875  3.68359375]
    #    [ 1.66015625  1.95703125  2.43359375  3.0625      3.46875
    #      4.09765625  4.57421875  4.87109375]
    #    [ 3.56640625  3.86328125  4.33984375  4.96875     5.375
    #      6.00390625  6.48046875  6.77734375]
    #    [ 6.08203125  6.37890625  6.85546875  7.484375    7.890625
    #      8.51953125  8.99609375  9.29296875]
    #    [ 7.70703125  8.00390625  8.48046875  9.109375    9.515625
    #     10.14453125 10.62109375 10.91796875]
    #    [10.22265625 10.51953125 10.99609375 11.625      12.03125
    #     12.66015625 13.13671875 13.43359375]
    #    [12.12890625 12.42578125 12.90234375 13.53125    13.9375
    #     14.56640625 15.04296875 15.33984375]
    #    [13.31640625 13.61328125 14.08984375 14.71875    15.125
    #     15.75390625 16.23046875 16.52734375]]]]
    output = interpolate_nd(
        data, cubic_coeffs, scale_factors=scales).astype(np.float32)

    expect(node, inputs=[data, scales], outputs=[output],
           name='test_resize_upsample_scales_cubic')

**resize_upsample_scales_cubic_align_corners**

::

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', 'scales'],
        outputs=['Y'],
        mode='cubic',
        coordinate_transformation_mode='align_corners'
    )

    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=np.float32)

    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    # [[[[ 1.          1.34110787  1.80029155  2.32944606  2.67055394
    #      3.19970845  3.65889213  4.        ]
    #    [ 2.36443149  2.70553936  3.16472303  3.69387755  4.03498542
    #      4.56413994  5.02332362  5.36443149]
    #    [ 4.20116618  4.54227405  5.00145773  5.53061224  5.87172012
    #      6.40087464  6.86005831  7.20116618]
    #    [ 6.31778426  6.65889213  7.1180758   7.64723032  7.98833819
    #      8.51749271  8.97667638  9.31778426]
    #    [ 7.68221574  8.02332362  8.48250729  9.01166181  9.35276968
    #      9.8819242  10.34110787 10.68221574]
    #    [ 9.79883382 10.13994169 10.59912536 11.12827988 11.46938776
    #     11.99854227 12.45772595 12.79883382]
    #    [11.63556851 11.97667638 12.43586006 12.96501458 13.30612245
    #     13.83527697 14.29446064 14.63556851]
    #    [13.         13.34110787 13.80029155 14.32944606 14.67055394
    #     15.19970845 15.65889213 16.        ]]]]
    output = interpolate_nd(
        data, cubic_coeffs, scale_factors=scales, coordinate_transformation_mode='align_corners').astype(np.float32)

    expect(node, inputs=[data, scales], outputs=[output],
           name='test_resize_upsample_scales_cubic_align_corners')

**resize_downsample_scales_cubic**

::

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', 'scales'],
        outputs=['Y'],
        mode='cubic',
    )

    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=np.float32)

    scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

    # [[[[ 1.47119141  2.78125     4.08251953]
    #    [ 6.71142578  8.02148438  9.32275391]
    #    [11.91650391 13.2265625  14.52783203]]]]
    output = interpolate_nd(
        data, cubic_coeffs, scale_factors=scales).astype(np.float32)

    expect(node, inputs=[data, scales], outputs=[output],
           name='test_resize_downsample_scales_cubic')

**resize_downsample_scales_cubic_align_corners**

::

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', 'scales'],
        outputs=['Y'],
        mode='cubic',
        coordinate_transformation_mode='align_corners'
    )

    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=np.float32)

    scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

    # [[[[ 1.          2.39519159  3.79038317]
    #    [ 6.58076634  7.97595793  9.37114951]
    #    [12.16153268 13.55672427 14.95191585]]]]
    output = interpolate_nd(
        data, cubic_coeffs, scale_factors=scales, coordinate_transformation_mode='align_corners').astype(np.float32)

    expect(node, inputs=[data, scales], outputs=[output],
           name='test_resize_downsample_scales_cubic_align_corners')

**resize_upsample_sizes_cubic**

::

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', '', 'sizes'],
        outputs=['Y'],
        mode='cubic',
    )

    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=np.float32)

    sizes = np.array([1, 1, 9, 10], dtype=np.int64)

    # [[[[ 0.45507922  0.64057922  0.97157922  1.42257922  1.90732922
    #      2.22332922  2.70807922  3.15907922  3.49007922  3.67557922]
    #    [ 1.39437963  1.57987963  1.91087963  2.36187963  2.84662963
    #      3.16262963  3.64737963  4.09837963  4.42937963  4.61487963]
    #    [ 2.95130693  3.13680693  3.46780693  3.91880693  4.40355693
    #      4.71955693  5.20430693  5.65530693  5.98630693  6.17180693]
    #    [ 5.20525069  5.39075069  5.72175069  6.17275069  6.65750069
    #      6.97350069  7.45825069  7.90925069  8.24025069  8.42575069]
    #    [ 6.88975     7.07525     7.40625     7.85725     8.342
    #      8.658       9.14275     9.59375     9.92475    10.11025   ]
    #    [ 8.57424931  8.75974931  9.09074931  9.54174931 10.02649931
    #     10.34249931 10.82724931 11.27824931 11.60924931 11.79474931]
    #    [10.82819307 11.01369307 11.34469307 11.79569307 12.28044307
    #     12.59644307 13.08119307 13.53219307 13.86319307 14.04869307]
    #    [12.38512037 12.57062037 12.90162037 13.35262037 13.83737037
    #     14.15337037 14.63812037 15.08912037 15.42012037 15.60562037]
    #    [13.32442078 13.50992078 13.84092078 14.29192078 14.77667078
    #     15.09267078 15.57742078 16.02842078 16.35942078 16.54492078]]]]
    output = interpolate_nd(
        data, cubic_coeffs, output_size=sizes).astype(np.float32)

    expect(node, inputs=[data, sizes], outputs=[output],
           name='test_resize_upsample_sizes_cubic')

**resize_downsample_sizes_cubic**

::

        node = onnx.helper.make_node(
            'Resize',
            inputs=['X', '', '', 'sizes'],
            outputs=['Y'],
            mode='cubic',
        )

        data = np.array([[[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]]], dtype=np.float32)

        sizes = np.array([1, 1, 3, 3], dtype=np.int64)

        # [[[[ 1.63078704  3.00462963  4.37847222]
        #    [ 7.12615741  8.5         9.87384259]
        #    [12.62152778 13.99537037 15.36921296]]]]
        output = interpolate_nd(
            data, cubic_coeffs, output_size=sizes).astype(np.float32)

        expect(node, inputs=[data, sizes], outputs=[output],
               name='test_resize_downsample_sizes_cubic')

    # TensorFlow v1 bicubic with half_pixel_centers=True

**resize_upsample_scales_cubic_A_n0p5_exclude_outside**

::

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', 'scales'],
        outputs=['Y'],
        mode='cubic',
        cubic_coeff_a=-0.5,
        exclude_outside=True
    )

    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=np.float32)

    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    # [[[[ 0.55882353  0.81494204  1.35698249  1.89705882  2.39705882
    #      2.93713516  3.47917561  3.73529412]
    #    [ 1.58329755  1.83941606  2.38145651  2.92153285  3.42153285
    #      3.96160918  4.50364964  4.75976814]
    #    [ 3.75145936  4.00757787  4.54961832  5.08969466  5.58969466
    #      6.12977099  6.67181144  6.92792995]
    #    [ 5.91176471  6.16788321  6.70992366  7.25        7.75
    #      8.29007634  8.83211679  9.08823529]
    #    [ 7.91176471  8.16788321  8.70992366  9.25        9.75
    #     10.29007634 10.83211679 11.08823529]
    #    [10.07207005 10.32818856 10.87022901 11.41030534 11.91030534
    #     12.45038168 12.99242213 13.24854064]
    #    [12.24023186 12.49635036 13.03839082 13.57846715 14.07846715
    #     14.61854349 15.16058394 15.41670245]
    #    [13.26470588 13.52082439 14.06286484 14.60294118 15.10294118
    #     15.64301751 16.18505796 16.44117647]]]]
    output = interpolate_nd(data, lambda x: cubic_coeffs(x, A=-0.5), scale_factors=scales,
                            exclude_outside=True).astype(np.float32)

    expect(node, inputs=[data, scales], outputs=[output],
           name='test_resize_upsample_scales_cubic_A_n0p5_exclude_outside')

**resize_downsample_scales_cubic_A_n0p5_exclude_outside**

::

        node = onnx.helper.make_node(
            'Resize',
            inputs=['X', '', 'scales'],
            outputs=['Y'],
            mode='cubic',
            cubic_coeff_a=-0.5,
            exclude_outside=True
        )

        data = np.array([[[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]]], dtype=np.float32)

        scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

        # [[[[ 1.36812675  2.6695014   4.0133367 ]
        #    [ 6.57362535  7.875       9.2188353 ]
        #    [11.94896657 13.25034122 14.59417652]]]]
        output = interpolate_nd(data, lambda x: cubic_coeffs(x, A=-0.5), scale_factors=scales,
                                exclude_outside=True).astype(np.float32)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_downsample_scales_cubic_A_n0p5_exclude_outside')

    # TensorFlow v1 bicubic with half_pixel_centers=False

**resize_upsample_scales_cubic_asymmetric**

::

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', 'scales'],
        outputs=['Y'],
        mode='cubic',
        coordinate_transformation_mode='asymmetric'
    )

    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=np.float32)

    scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

    # [[[[ 1.       1.40625  2.       2.5      3.       3.59375  4.
    #      4.09375]
    #    [ 2.625    3.03125  3.625    4.125    4.625    5.21875  5.625
    #      5.71875]
    #    [ 5.       5.40625  6.       6.5      7.       7.59375  8.
    #      8.09375]
    #    [ 7.       7.40625  8.       8.5      9.       9.59375 10.
    #     10.09375]
    #    [ 9.       9.40625 10.      10.5     11.      11.59375 12.
    #     12.09375]
    #    [11.375   11.78125 12.375   12.875   13.375   13.96875 14.375
    #     14.46875]
    #    [13.      13.40625 14.      14.5     15.      15.59375 16.
    #     16.09375]
    #    [13.375   13.78125 14.375   14.875   15.375   15.96875 16.375
    #     16.46875]]]]
    output = interpolate_nd(data, lambda x: cubic_coeffs(x, A=-0.75), scale_factors=scales,
                            coordinate_transformation_mode='asymmetric').astype(np.float32)

    expect(node, inputs=[data, scales], outputs=[output],
           name='test_resize_upsample_scales_cubic_asymmetric')

**resize_tf_crop_and_resize**

::

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', 'roi', '', 'sizes'],
        outputs=['Y'],
        mode='linear',
        coordinate_transformation_mode='tf_crop_and_resize'
    )

    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=np.float32)

    # Note: for some rois, the result may be different with that of TF for inaccurate floating point
    roi = np.array([0, 0, 0.4, 0.6, 1, 1, 0.6, 0.8], dtype=np.float32)
    sizes = np.array([1, 1, 3, 3], dtype=np.int64)

    # [[[[ 7.6000004  7.9        8.2      ]
    #    [ 8.8        9.1        9.400001 ]
    #    [10.        10.3       10.6      ]]]]
    output = interpolate_nd(data, linear_coeffs, output_size=sizes, roi=roi,
                            coordinate_transformation_mode='tf_crop_and_resize').astype(np.float32)

    expect(node, inputs=[data, roi, sizes], outputs=[output],
           name='test_resize_tf_crop_and_resize')

**resize_tf_crop_and_resize_extrapolation_value**

::

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', 'roi', '', 'sizes'],
        outputs=['Y'],
        mode='linear',
        coordinate_transformation_mode='tf_crop_and_resize',
        extrapolation_value=10.0
    )

    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=np.float32)

    # Note: for some rois, the result may be different with that of TF for inaccurate floating point
    roi = np.array([0, 0, 0.4, 0.6, 1, 1, 1.2, 1.7], dtype=np.float32)
    sizes = np.array([1, 1, 3, 3], dtype=np.int64)

    # [[[[ 7.6000004 10.        10.       ]
    #    [12.400001  10.        10.       ]
    #    [10.        10.        10.       ]]]]
    output = interpolate_nd(data, linear_coeffs, output_size=sizes, roi=roi,
                            coordinate_transformation_mode='tf_crop_and_resize', extrapolation_value=10.0).astype(np.float32)

    expect(node, inputs=[data, roi, sizes], outputs=[output],
           name='test_resize_tf_crop_and_resize')

**resize_downsample_sizes_linear_pytorch_half_pixel**

::

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', '', 'sizes'],
        outputs=['Y'],
        mode='linear',
        coordinate_transformation_mode='pytorch_half_pixel'
    )

    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=np.float32)

    sizes = np.array([1, 1, 3, 1], dtype=np.int64)

    # [[[[ 1.6666666]
    #    [ 7.       ]
    #    [12.333333 ]]]]
    output = interpolate_nd(
        data, linear_coeffs, output_size=sizes, coordinate_transformation_mode='pytorch_half_pixel').astype(np.float32)

    expect(node, inputs=[data, sizes], outputs=[output],
           name='test_resize_downsample_sizes_linear_pytorch_half_pixel')

**resize_upsample_sizes_nearest_floor_align_corners**

::

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', '', 'sizes'],
        outputs=['Y'],
        mode='nearest',
        coordinate_transformation_mode='align_corners',
        nearest_mode='floor'
    )

    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=np.float32)

    sizes = np.array([1, 1, 8, 8], dtype=np.int64)

    # [[[[ 1.  1.  1.  2.  2.  3.  3.  4.]
    #    [ 1.  1.  1.  2.  2.  3.  3.  4.]
    #    [ 1.  1.  1.  2.  2.  3.  3.  4.]
    #    [ 5.  5.  5.  6.  6.  7.  7.  8.]
    #    [ 5.  5.  5.  6.  6.  7.  7.  8.]
    #    [ 9.  9.  9. 10. 10. 11. 11. 12.]
    #    [ 9.  9.  9. 10. 10. 11. 11. 12.]
    #    [13. 13. 13. 14. 14. 15. 15. 16.]]]]
    output = interpolate_nd(
        data, lambda x: nearest_coeffs(x, mode='floor'), output_size=sizes, coordinate_transformation_mode='align_corners').astype(np.float32)

    expect(node, inputs=[data, sizes], outputs=[output],
           name='test_resize_upsample_sizes_nearest_floor_align_corners')

**resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric**

::

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', '', 'sizes'],
        outputs=['Y'],
        mode='nearest',
        coordinate_transformation_mode='asymmetric',
        nearest_mode='round_prefer_ceil'
    )

    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=np.float32)

    sizes = np.array([1, 1, 8, 8], dtype=np.int64)

    # [[[[ 1.  2.  2.  3.  3.  4.  4.  4.]
    #    [ 5.  6.  6.  7.  7.  8.  8.  8.]
    #    [ 5.  6.  6.  7.  7.  8.  8.  8.]
    #    [ 9. 10. 10. 11. 11. 12. 12. 12.]
    #    [ 9. 10. 10. 11. 11. 12. 12. 12.]
    #    [13. 14. 14. 15. 15. 16. 16. 16.]
    #    [13. 14. 14. 15. 15. 16. 16. 16.]
    #    [13. 14. 14. 15. 15. 16. 16. 16.]]]]
    output = interpolate_nd(
        data, lambda x: nearest_coeffs(x, mode='round_prefer_ceil'),
        output_size=sizes, coordinate_transformation_mode='asymmetric').astype(np.float32)

    expect(node, inputs=[data, sizes], outputs=[output],
           name='test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric')

**resize_upsample_sizes_nearest_ceil_half_pixel**

::

    node = onnx.helper.make_node(
        'Resize',
        inputs=['X', '', '', 'sizes'],
        outputs=['Y'],
        mode='nearest',
        coordinate_transformation_mode='half_pixel',
        nearest_mode='ceil'
    )

    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=np.float32)

    sizes = np.array([1, 1, 8, 8], dtype=np.int64)

    # [[[[ 1.  2.  2.  3.  3.  4.  4.  4.]
    #    [ 5.  6.  6.  7.  7.  8.  8.  8.]
    #    [ 5.  6.  6.  7.  7.  8.  8.  8.]
    #    [ 9. 10. 10. 11. 11. 12. 12. 12.]
    #    [ 9. 10. 10. 11. 11. 12. 12. 12.]
    #    [13. 14. 14. 15. 15. 16. 16. 16.]
    #    [13. 14. 14. 15. 15. 16. 16. 16.]
    #    [13. 14. 14. 15. 15. 16. 16. 16.]]]]
    output = interpolate_nd(
        data, lambda x: nearest_coeffs(x, mode='ceil'), output_size=sizes).astype(np.float32)

    expect(node, inputs=[data, sizes], outputs=[output],
           name='test_resize_upsample_sizes_nearest_ceil_half_pixel')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Each dimension value of the output tensor is:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Each dimension value of the output tensor is:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output_dimension = floor(input_dimension * (roi_end - roi_start) * scale) if input \"sizes\" is not specified.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output_dimension = floor(input_dimension * (roi_end - roi_start) * scale) if input \"sizes\" is not specified.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **coordinate_transformation_mode**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **coordinate_transformation_mode**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   This attribute describes how to transform the coordinate in the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   This attribute describes how to transform the coordinate in the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  resized tensor to the coordinate in the original tensor. <br/>  The</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  resized tensor to the coordinate in the original tensor. <br/>  The</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  coordinate of each dimension is transformed individually. Let's</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  coordinate of each dimension is transformed individually. Let's</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  describe a case using axis x as an example. Denote x_resized as the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  describe a case using axis x as an example. Denote x_resized as the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  coordinate of axis x in the resized tensor, x_original as the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  coordinate of axis x in the resized tensor, x_original as the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  coordinate of axis x in the original tensor, length_original as the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  coordinate of axis x in the original tensor, length_original as the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  length of the original tensor in axis x, length_resized as the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  length of the original tensor in axis x, length_resized as the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  length of the resized tensor in axis x, roi_x = (start_x, end_x) of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  length of the resized tensor in axis x, roi_x = (start_x, end_x) of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the axis x in input "roi", scale = length_resized / length_original,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the axis x in input "roi", scale = length_resized / length_original,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  <br/>  if coordinate_transformation_mode is "half_pixel", <br/></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  <br/>  if coordinate_transformation_mode is "half_pixel", <br/></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x_original = (x_resized + 0.5) / scale - 0.5, <br/>  if</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x_original = (x_resized + 0.5) / scale - 0.5, <br/>  if</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  coordinate_transformation_mode is "pytorch_half_pixel", <br/></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  coordinate_transformation_mode is "pytorch_half_pixel", <br/></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 :</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 :</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  0, <br/>  if coordinate_transformation_mode is "align_corners",</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  0, <br/>  if coordinate_transformation_mode is "align_corners",</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  <br/> x_original = x_resized * (length_original - 1) /</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  <br/> x_original = x_resized * (length_original - 1) /</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (length_resized - 1), <br/>  if coordinate_transformation_mode is</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (length_resized - 1), <br/>  if coordinate_transformation_mode is</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  "asymmetric", <br/> x_original = x_resized / scale, <br/>  if</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  "asymmetric", <br/> x_original = x_resized / scale, <br/>  if</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">24</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  coordinate_transformation_mode is "tf_half_pixel_for_nn", <br/></code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">25</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  x_original = (x_resized + 0.5) / scale, <br/>  if</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  coordinate_transformation_mode is "tf_crop_and_resize", <br/></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  coordinate_transformation_mode is "tf_crop_and_resize", <br/></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x_original = length_resized > 1 ? start_x * (length_original - 1) +</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x_original = length_resized > 1 ? start_x * (length_original - 1) +</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x_resized * (end_x - start_x) * (length_original - 1) /</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x_resized * (end_x - start_x) * (length_original - 1) /</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original -</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original -</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  1). Default value is 'half_pixel'.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  1). Default value is 'half_pixel'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **cubic_coeff_a**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **cubic_coeff_a**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The coefficient 'a' used in cubic interpolation. Two common choice</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The coefficient 'a' used in cubic interpolation. Two common choice</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  are -0.5 (in some cases of TensorFlow) and -0.75 (in PyTorch). Check</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  are -0.5 (in some cases of TensorFlow) and -0.75 (in PyTorch). Check</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  out Equation (4) in https://ieeexplore.ieee.org/document/1163711 for</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  out Equation (4) in https://ieeexplore.ieee.org/document/1163711 for</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the details. This attribute is valid only if "mode" is "cubic". Default value is -0.75.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the details. This attribute is valid only if "mode" is "cubic". Default value is -0.75.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **exclude_outside**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **exclude_outside**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  If set to 1, the weight of sampling locations outside the tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  If set to 1, the weight of sampling locations outside the tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  will be set to 0 and the weight will be renormalized so that their</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  will be set to 0 and the weight will be renormalized so that their</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  sum is 1.0. The default value is 0. Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  sum is 1.0. The default value is 0. Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **extrapolation_value**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **extrapolation_value**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  When coordinate_transformation_mode is "tf_crop_and_resize" and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  When coordinate_transformation_mode is "tf_crop_and_resize" and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x_original is outside the range [0, length_original - 1], this value</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x_original is outside the range [0, length_original - 1], this value</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  is used as the corresponding output value. Default is 0.0f. Default value is 0.0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  is used as the corresponding output value. Default is 0.0f. Default value is 0.0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **mode**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **mode**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Three interpolation modes: nearest (default), linear and cubic. The</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Three interpolation modes: nearest (default), linear and cubic. The</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  "linear" mode includes linear interpolation for 1D tensor and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  "linear" mode includes linear interpolation for 1D tensor and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  N-linear interpolation for N-D tensor (for example, bilinear</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  N-linear interpolation for N-D tensor (for example, bilinear</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  interpolation for 2D tensor). The "cubic" mode includes cubic</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  interpolation for 2D tensor). The "cubic" mode includes cubic</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  interpolation for 1D tensor and N-cubic interpolation for N-D tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  interpolation for 1D tensor and N-cubic interpolation for N-D tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (for example, bicubic interpolation for 2D tensor). Default value is 'nearest'.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (for example, bicubic interpolation for 2D tensor). Default value is 'nearest'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nearest_mode**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nearest_mode**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Four modes: round_prefer_floor (default, as known as round half</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Four modes: round_prefer_floor (default, as known as round half</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  down), round_prefer_ceil (as known as round half up), floor, ceil.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  down), round_prefer_ceil (as known as round half up), floor, ceil.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Only used by nearest interpolation. It indicates how to get</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Only used by nearest interpolation. It indicates how to get</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  "nearest" pixel in input tensor from x_original, so this attribute</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  "nearest" pixel in input tensor from x_original, so this attribute</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  is valid only if "mode" is "nearest". Default value is 'round_prefer_floor'.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  is valid only if "mode" is "nearest". Default value is 'round_prefer_floor'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td><code>60</code></td><td><code>58</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">Between <span style="color:#BA4A00;">3</span> and 4 inputs.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>Between <span style="color:#196F3D;">1</span> and 4 inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T1**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T1**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  N-D tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  N-D tensor</code></td></tr>
    <tr style="1px solid black;"><td><code>64</code></td><td><code>62</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">* **roi** (heterogeneous) - **T2**:</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>* **roi** (<span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span>heterogeneous) - **T2**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  1-D tensor given as [start1, ..., startN, end1, ..., endN], where N</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  1-D tensor given as [start1, ..., startN, end1, ..., endN], where N</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  is the rank of X. The RoIs' coordinates are normalized in the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  is the rank of X. The RoIs' coordinates are normalized in the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  coordinate system of the input image. It only takes effect when</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  coordinate system of the input image. It only takes effect when</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  coordinate_transformation_mode is "tf_crop_and_resize"</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  coordinate_transformation_mode is "tf_crop_and_resize"</code></td></tr>
    <tr style="1px solid black;"><td><code>69</code></td><td><code>67</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">* **scales** (heterogeneous) - **tensor(float)**:</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>* **scales** (<span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span>heterogeneous) - **tensor(float)**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The scale array along each dimension. It takes value greater than 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The scale array along each dimension. It takes value greater than 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  If it's less than 1, it's sampling down, otherwise, it's upsampling.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  If it's less than 1, it's sampling down, otherwise, it's upsampling.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The number of elements of 'scales' should be the same as the rank of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The number of elements of 'scales' should be the same as the rank of</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">71</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  input 'X'. One of 'scales' and 'sizes' MUST be specified and it is</code></td></tr>
    <tr style="1px solid black;"><td><code>73</code></td><td><code>72</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  i<span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">u</span>t <span style="color:#BA4A00;">'</span><span style="color:#BA4A00;">X</span><span style="color:#BA4A00;">'</span>. If 'size' is needed, the user <span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">'</span><span style="color:#BA4A00;">s</span>ca<span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">'</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span>n</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span>i<span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">o</span>t<span style="color:#196F3D;">h</span> <span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span>. If 'size<span style="color:#196F3D;">s</span>' is needed, the user can</code></td></tr>
    <tr style="1px solid black;"><td><code>74</code></td><td><code>73</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  empty tensor<span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">u</span><span style="color:#196F3D;">s</span>e<span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">e</span>mpty <span style="color:#196F3D;">s</span>t<span style="color:#196F3D;">r</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span>e<span style="color:#196F3D;"> </span>n<span style="color:#196F3D;">a</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">'</span>s<span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">'</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span>o<span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span>r<span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">'</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">74</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  list.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **sizes** (optional, heterogeneous) - **tensor(int64)**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **sizes** (optional, heterogeneous) - **tensor(int64)**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The size of the output tensor. The number of elements of 'sizes'</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The size of the output tensor. The number of elements of 'sizes'</code></td></tr>
    <tr style="1px solid black;"><td><code>77</code></td><td><code>77</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  should be the same as the rank of input 'X'. <span style="color:#BA4A00;">M</span><span style="color:#BA4A00;">a</span>y on<span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">y</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">b</span>e se<span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">f</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  should be the same as the rank of input 'X'. <span style="color:#196F3D;">O</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">l</span>y one <span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">'</span>s<span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span>e<span style="color:#196F3D;">s</span><span style="color:#196F3D;">'</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">78</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  'scales' is set to an empty tensor.</code></td><td></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">78</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  and 'sizes' can be specified.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T1**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T1**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  N-D tensor after resizing</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  N-D tensor after resizing</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">88</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(bfloat16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input 'X' and output 'Y' to all tensor types.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input 'X' and output 'Y' to all tensor types.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T2** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T2** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">109</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">109</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">110</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">110</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">111</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain roi type to float or double.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain roi type to float or double.</code></td></tr>
    </table>

.. _l-onnx-op-resize-11:

Resize - 11
===========

**Version**

* **name**: `Resize (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * (roi_end - roi_start) * scale) if input \"sizes\" is not specified.

**Attributes**

* **coordinate_transformation_mode**:
   This attribute describes how to transform the coordinate in the
  resized tensor to the coordinate in the original tensor. <br/>  The
  coordinate of each dimension is transformed individually. Let's
  describe a case using axis x as an example. Denote x_resized as the
  coordinate of axis x in the resized tensor, x_original as the
  coordinate of axis x in the original tensor, length_original as the
  length of the original tensor in axis x, length_resized as the
  length of the resized tensor in axis x, roi_x = (start_x, end_x) of
  the axis x in input "roi", scale = length_resized / length_original,
  <br/>  if coordinate_transformation_mode is "half_pixel", <br/>
  x_original = (x_resized + 0.5) / scale - 0.5, <br/>  if
  coordinate_transformation_mode is "pytorch_half_pixel", <br/>
  x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 :
  0, <br/>  if coordinate_transformation_mode is "align_corners",
  <br/> x_original = x_resized * (length_original - 1) /
  (length_resized - 1), <br/>  if coordinate_transformation_mode is
  "asymmetric", <br/> x_original = x_resized / scale, <br/>  if
  coordinate_transformation_mode is "tf_half_pixel_for_nn", <br/>
  x_original = (x_resized + 0.5) / scale, <br/>  if
  coordinate_transformation_mode is "tf_crop_and_resize", <br/>
  x_original = length_resized > 1 ? start_x * (length_original - 1) +
  x_resized * (end_x - start_x) * (length_original - 1) /
  (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original -
  1). Default value is ``'half_pixel'``.
* **cubic_coeff_a**:
  The coefficient 'a' used in cubic interpolation. Two common choice
  are -0.5 (in some cases of TensorFlow) and -0.75 (in PyTorch). Check
  out Equation (4) in https://ieeexplore.ieee.org/document/1163711 for
  the details. This attribute is valid only if "mode" is "cubic". Default value is ``-0.75``.
* **exclude_outside**:
  If set to 1, the weight of sampling locations outside the tensor
  will be set to 0 and the weight will be renormalized so that their
  sum is 1.0. The default value is 0. Default value is ``0``.
* **extrapolation_value**:
  When coordinate_transformation_mode is "tf_crop_and_resize" and
  x_original is outside the range [0, length_original - 1], this value
  is used as the corresponding output value. Default is 0.0f. Default value is ``0.0``.
* **mode**:
  Three interpolation modes: nearest (default), linear and cubic. The
  "linear" mode includes linear interpolation for 1D tensor and
  N-linear interpolation for N-D tensor (for example, bilinear
  interpolation for 2D tensor). The "cubic" mode includes cubic
  interpolation for 1D tensor and N-cubic interpolation for N-D tensor
  (for example, bicubic interpolation for 2D tensor). Default value is ``'nearest'``.
* **nearest_mode**:
  Four modes: round_prefer_floor (default, as known as round half
  down), round_prefer_ceil (as known as round half up), floor, ceil.
  Only used by nearest interpolation. It indicates how to get
  "nearest" pixel in input tensor from x_original, so this attribute
  is valid only if "mode" is "nearest". Default value is ``'round_prefer_floor'``.

**Inputs**

Between 3 and 4 inputs.

* **X** (heterogeneous) - **T1**:
  N-D tensor
* **roi** (heterogeneous) - **T2**:
  1-D tensor given as [start1, ..., startN, end1, ..., endN], where N
  is the rank of X. The RoIs' coordinates are normalized in the
  coordinate system of the input image. It only takes effect when
  coordinate_transformation_mode is "tf_crop_and_resize"
* **scales** (heterogeneous) - **tensor(float)**:
  The scale array along each dimension. It takes value greater than 0.
  If it's less than 1, it's sampling down, otherwise, it's upsampling.
  The number of elements of 'scales' should be the same as the rank of
  input 'X'. If 'size' is needed, the user must set 'scales' to an
  empty tensor.
* **sizes** (optional, heterogeneous) - **tensor(int64)**:
  The size of the output tensor. The number of elements of 'sizes'
  should be the same as the rank of input 'X'. May only be set if
  'scales' is set to an empty tensor.

**Outputs**

* **Y** (heterogeneous) - **T1**:
  N-D tensor after resizing

**Type Constraints**

* **T1** in (
  tensor(bool),
  tensor(complex128),
  tensor(complex64),
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int16),
  tensor(int32),
  tensor(int64),
  tensor(int8),
  tensor(string),
  tensor(uint16),
  tensor(uint32),
  tensor(uint64),
  tensor(uint8)
  ):
  Constrain input 'X' and output 'Y' to all tensor types.
* **T2** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain roi type to float or double.

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td><code>0</code></td><td><code>0</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">Resize the input tensor.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>Resize the input tensor.<span style="color:#196F3D;"> </span><span style="color:#196F3D;">I</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">w</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">k</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Each dimension value of the output tensor is:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Each dimension value of the output tensor is:</code></td></tr>
    <tr style="1px solid black;"><td><code>2</code></td><td><code>2</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  output_dimension = floor(input_dimension * scale).</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  output_dimension = floor(input_dimension * <span style="color:#196F3D;">(</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">_</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">-</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">_</span>s<span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">*</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span>cale)<span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">\</span><span style="color:#196F3D;">"</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">z</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">\</span><span style="color:#196F3D;">"</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span>.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">6</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **coordinate_transformation_mode**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">7</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">   This attribute describes how to transform the coordinate in the</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">8</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  resized tensor to the coordinate in the original tensor. <br/>  The</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">9</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  coordinate of each dimension is transformed individually. Let's</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">10</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  describe a case using axis x as an example. Denote x_resized as the</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">11</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  coordinate of axis x in the resized tensor, x_original as the</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">12</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  coordinate of axis x in the original tensor, length_original as the</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">13</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  length of the original tensor in axis x, length_resized as the</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">14</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  length of the resized tensor in axis x, roi_x = (start_x, end_x) of</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">15</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  the axis x in input "roi", scale = length_resized / length_original,</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">16</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  <br/>  if coordinate_transformation_mode is "half_pixel", <br/></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">17</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  x_original = (x_resized + 0.5) / scale - 0.5, <br/>  if</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">18</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  coordinate_transformation_mode is "pytorch_half_pixel", <br/></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">19</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 :</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">20</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  0, <br/>  if coordinate_transformation_mode is "align_corners",</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">21</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  <br/> x_original = x_resized * (length_original - 1) /</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">22</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  (length_resized - 1), <br/>  if coordinate_transformation_mode is</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">23</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  "asymmetric", <br/> x_original = x_resized / scale, <br/>  if</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">24</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  coordinate_transformation_mode is "tf_half_pixel_for_nn", <br/></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">25</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  x_original = (x_resized + 0.5) / scale, <br/>  if</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">26</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  coordinate_transformation_mode is "tf_crop_and_resize", <br/></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">27</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  x_original = length_resized > 1 ? start_x * (length_original - 1) +</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">28</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  x_resized * (end_x - start_x) * (length_original - 1) /</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">29</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original -</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">30</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  1). Default value is 'half_pixel'.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">31</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **cubic_coeff_a**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">32</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  The coefficient 'a' used in cubic interpolation. Two common choice</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">33</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  are -0.5 (in some cases of TensorFlow) and -0.75 (in PyTorch). Check</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">34</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  out Equation (4) in https://ieeexplore.ieee.org/document/1163711 for</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">35</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  the details. This attribute is valid only if "mode" is "cubic". Default value is -0.75.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">36</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **exclude_outside**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">37</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  If set to 1, the weight of sampling locations outside the tensor</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">38</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  will be set to 0 and the weight will be renormalized so that their</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">39</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  sum is 1.0. The default value is 0. Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">40</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **extrapolation_value**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">41</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  When coordinate_transformation_mode is "tf_crop_and_resize" and</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">42</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  x_original is outside the range [0, length_original - 1], this value</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">43</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  is used as the corresponding output value. Default is 0.0f. Default value is 0.0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **mode**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **mode**:</code></td></tr>
    <tr style="1px solid black;"><td><code>7</code></td><td><code>45</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  T<span style="color:#BA4A00;">w</span><span style="color:#BA4A00;">o</span> interpolation modes: nearest (default), <span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;"> </span>linear <span style="color:#BA4A00;">(</span><span style="color:#BA4A00;">i</span>nc<span style="color:#BA4A00;">l</span>u<span style="color:#BA4A00;">d</span>i<span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">g</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  T<span style="color:#196F3D;">h</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">e</span> interpolation modes: nearest (default), linear <span style="color:#196F3D;">a</span>n<span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span>cu<span style="color:#196F3D;">b</span>i<span style="color:#196F3D;">c</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">T</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">46</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  "linear" mode includes linear interpolation for 1D tensor and</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">47</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  N-linear interpolation for N-D tensor (for example, bilinear</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">48</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  interpolation for 2D tensor). The "cubic" mode includes cubic</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">49</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  interpolation for 1D tensor and N-cubic interpolation for N-D tensor</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">50</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  (for example, bicubic interpolation for 2D tensor). Default value is 'nearest'.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">51</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **nearest_mode**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">52</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Four modes: round_prefer_floor (default, as known as round half</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">53</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  down), round_prefer_ceil (as known as round half up), floor, ceil.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">54</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Only used by nearest interpolation. It indicates how to get</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">55</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  "nearest" pixel in input tensor from x_original, so this attribute</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">56</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  is valid only if "mode" is "nearest". Default value is 'round_prefer_floor'.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">57</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">58</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">59</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td><code>8</code></td><td><code>60</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span>e<span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span>t<span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span>e<span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span>e<span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">)</span> <span style="color:#BA4A00;">D</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">t</span> <span style="color:#BA4A00;">v</span>a<span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">e</span> i<span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">'</span>n<span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">e</span>s<span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">'</span>.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">B</span>et<span style="color:#196F3D;">w</span>ee<span style="color:#196F3D;">n</span> <span style="color:#196F3D;">3</span> a<span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span> <span style="color:#196F3D;">4</span><span style="color:#196F3D;"> </span>in<span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span>s.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td><code>10</code></td><td><code>62</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">**<span style="color:#BA4A00;">I</span>n<span style="color:#BA4A00;">p</span>u<span style="color:#BA4A00;">t</span>s**</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>*<span style="color:#196F3D;"> </span>*<span style="color:#196F3D;">*</span><span style="color:#196F3D;">X</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">e</span>n<span style="color:#196F3D;">e</span><span style="color:#196F3D;">o</span>us<span style="color:#196F3D;">)</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">-</span><span style="color:#196F3D;"> </span>**<span style="color:#196F3D;">T</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">:</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">11</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;"></code></td><td></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">63</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  N-D tensor</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">64</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **roi** (heterogeneous) - **T2**:</code></td></tr>
    <tr style="1px solid black;"><td><code>12</code></td><td><code>65</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">*</span> <span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">X</span><span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">*</span> <span style="color:#BA4A00;">(</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span>ter<span style="color:#BA4A00;">o</span>gen<span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">u</span>s<span style="color:#BA4A00;">)</span> <span style="color:#BA4A00;">-</span> <span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">T</span><span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">:</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">1</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">D</span><span style="color:#196F3D;"> </span>te<span style="color:#196F3D;">n</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">o</span>r<span style="color:#196F3D;"> </span>g<span style="color:#196F3D;">i</span><span style="color:#196F3D;">v</span>en<span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span>s <span style="color:#196F3D;">[</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">,</span> <span style="color:#196F3D;">.</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">N</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">N</span><span style="color:#196F3D;">]</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">w</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">N</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">66</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  is the rank of X. The RoIs' coordinates are normalized in the</code></td></tr>
    <tr style="1px solid black;"><td><code>13</code></td><td><code>67</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  <span style="color:#BA4A00;">N</span><span style="color:#BA4A00;">-</span><span style="color:#BA4A00;">D</span> tens<span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">c</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span> <span style="color:#196F3D;">s</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;">s</span>te<span style="color:#196F3D;">m</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span>n<span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">I</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">k</span><span style="color:#196F3D;">e</span>s<span style="color:#196F3D;"> </span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">w</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">68</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  coordinate_transformation_mode is "tf_crop_and_resize"</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **scales** (heterogeneous) - **tensor(float)**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **scales** (heterogeneous) - **tensor(float)**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The scale array along each dimension. It takes value greater than 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The scale array along each dimension. It takes value greater than 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  If it's less than 1, it's sampling down, otherwise, it's upsampling.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  If it's less than 1, it's sampling down, otherwise, it's upsampling.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The number of elements of 'scales' should be the same as the rank of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The number of elements of 'scales' should be the same as the rank of</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">73</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  input 'X'. If 'size' is needed, the user must set 'scales' to an</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">74</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  empty tensor.</code></td></tr>
    <tr style="1px solid black;"><td><code>18</code></td><td><code>75</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  in<span style="color:#BA4A00;">p</span>ut<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">'</span><span style="color:#BA4A00;">X</span><span style="color:#BA4A00;">'</span><span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">*</span> <span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">z</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span> <span style="color:#196F3D;">(</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span>i<span style="color:#196F3D;">o</span>n<span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">o</span>u<span style="color:#196F3D;">s</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">-</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span>t<span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">6</span><span style="color:#196F3D;">4</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">:</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">76</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  The size of the output tensor. The number of elements of 'sizes'</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">77</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  should be the same as the rank of input 'X'. May only be set if</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">78</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  'scales' is set to an empty tensor.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td><code>22</code></td><td><code>82</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">* **Y** (heterogeneous) - **T**:</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>* **Y** (heterogeneous) - **T<span style="color:#196F3D;">1</span>**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  N-D tensor after resizing</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  N-D tensor after resizing</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td><code>27</code></td><td><code>87</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">* **T** in (</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>* **T<span style="color:#196F3D;">1</span>** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input 'X' and output 'Y' to all tensor types.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input 'X' and output 'Y' to all tensor types.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">105</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **T2** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">106</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">107</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">108</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">109</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  ):</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">110</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Constrain roi type to float or double.</code></td></tr>
    </table>

.. _l-onnx-op-resize-10:

Resize - 10
===========

**Version**

* **name**: `Resize (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize>`_
* **domain**: **main**
* **since_version**: **10**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 10**.

**Summary**

Resize the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).

**Attributes**

* **mode**:
  Two interpolation modes: nearest (default), and linear (including
  bilinear, trilinear, etc) Default value is ``'nearest'``.

**Inputs**

* **X** (heterogeneous) - **T**:
  N-D tensor
* **scales** (heterogeneous) - **tensor(float)**:
  The scale array along each dimension. It takes value greater than 0.
  If it's less than 1, it's sampling down, otherwise, it's upsampling.
  The number of elements of 'scales' should be the same as the rank of
  input 'X'.

**Outputs**

* **Y** (heterogeneous) - **T**:
  N-D tensor after resizing

**Type Constraints**

* **T** in (
  tensor(bool),
  tensor(complex128),
  tensor(complex64),
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int16),
  tensor(int32),
  tensor(int64),
  tensor(int8),
  tensor(string),
  tensor(uint16),
  tensor(uint32),
  tensor(uint64),
  tensor(uint8)
  ):
  Constrain input 'X' and output 'Y' to all tensor types.
