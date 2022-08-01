
.. _l-onnx-doc-GridSample:

==========
GridSample
==========

.. contents::
    :local:


.. _l-onnx-op-gridsample-16:

GridSample - 16
===============

**Version**

* **name**: `GridSample (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#GridSample>`_
* **domain**: **main**
* **since_version**: **16**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 16**.

**Summary**

Given an `input` and a flow-field `grid`, computes the `output` using `input` values and pixel locations from `grid`.
Currently, only spatial (4-D) inputs are supported. For `input` with shape (N, C, H, W) and `grid` with shape (N, H_out, W_out, 2),
the `output` will have shape (N, C, H_out, W_out).
For each output location `output[N, C, H_out, W_out]`, the size-2 vector `grid[N, H_out, W_out]` specifies `input` pixel locations `x` and `y`,
which are used to interpolate the output value `output[N, C, H_out, W_out]`.

The GridSample operator is often used in doing grid generator and sampler in the [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025).
See also in [torch.nn.functional.grid_sample](https://pytorch.org/docs/master/generated/torch.nn.functional.grid_sample.html#torch-nn-functional-grid-sample).

**Attributes**

* **align_corners**:
  If align_corners=1, the extrema (-1 and 1) are considered as
  referring to the center points of the input's corner pixels. If
  align_corners=0, they are instead considered as referring to the
  corner points of the input's corner pixels, making the sampling more
  resolution agnostic. Default value is ``0``.
* **mode**:
  Three interpolation modes: bilinear (default), nearest and bicubic. Default value is ``'bilinear'``.
* **padding_mode**:
  Support padding modes for outside grid values: `zeros`(default),
  `border`, `reflection`. zeros: use 0 for out-of-bound grid
  locations, border: use border values for out-of-bound grid
  locations, reflection: use values at locations reflected by the
  border for out-of-bound grid locations. If index 0 represents the
  margin pixel, the reflected value at index -1 will be the same as
  the value at index 1. For location far away from the border, it will
  keep being reflected until becoming in bound. If pixel location x =
  -3.5 reflects by border -1 and becomes x' = 1.5, then reflects by
  border 1 and becomes x'' = 0.5. Default value is ``'zeros'``.

**Inputs**

* **X** (heterogeneous) - **T1**:
  4-D tensor of shape (N, C, H, W), where N is the batch size, C is
  the numbers of channels, H and W are the height and width of the
  input data.
* **grid** (heterogeneous) - **T1**:
  Input offset, 4-D tensor of shape (N, H_out, W_out, 2), where H_out
  and W_out are the height and width of grid and output, Grid
  specifies the sampling pixel locations normalized by the input
  spatial dimensions. Therefore, it should have most values in the
  range of [-1, 1]. If grid has values outside the range of [-1, 1],
  the corresponding outputs will be handled as defined by
  padding_mode.

**Outputs**

* **Y** (heterogeneous) - **T2**:
  4-D tensor of shape (N, C, H_out, W_out).

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
  Constrain input types to all tensor types.
* **T2** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain output types to float tensors.

**Examples**

**gridsample**

::

    node = onnx.helper.make_node(
        'GridSample',
        inputs=['X', 'Grid'],
        outputs=['Y'],
        mode='bilinear',
        padding_mode='zeros',
        align_corners=0,
    )
    # X shape, [N, C, H, W] - [1, 1, 4, 4]
    X = np.array(
        [
            [
                [
                    [0., 1., 2., 3.],
                    [4., 5., 6., 7.],
                    [8., 9., 10., 11.],
                    [12., 13., 14., 15.]
                ]
            ]
        ],
        dtype=np.float32,
    )
    # Grid shape, [N, H_out, W_out, 2] - [1, 6, 6, 2]
    Grid = np.array(
        [
            [
                [
                    [-1.0000, -1.0000],
                    [-0.6000, -1.0000],
                    [-0.2000, -1.0000],
                    [0.2000, -1.0000],
                    [0.6000, -1.0000],
                    [1.0000, -1.0000]
                ],
                [
                    [-1.0000, -0.6000],
                    [-0.6000, -0.6000],
                    [-0.2000, -0.6000],
                    [0.2000, -0.6000],
                    [0.6000, -0.6000],
                    [1.0000, -0.6000]
                ],
                [
                    [-1.0000, -0.2000],
                    [-0.6000, -0.2000],
                    [-0.2000, -0.2000],
                    [0.2000, -0.2000],
                    [0.6000, -0.2000],
                    [1.0000, -0.2000]
                ],
                [
                    [-1.0000, 0.2000],
                    [-0.6000, 0.2000],
                    [-0.2000, 0.2000],
                    [0.2000, 0.2000],
                    [0.6000, 0.2000],
                    [1.0000, 0.2000]
                ],
                [
                    [-1.0000, 0.6000],
                    [-0.6000, 0.6000],
                    [-0.2000, 0.6000],
                    [0.2000, 0.6000],
                    [0.6000, 0.6000],
                    [1.0000, 0.6000]
                ],
                [
                    [-1.0000, 1.0000],
                    [-0.6000, 1.0000],
                    [-0.2000, 1.0000],
                    [0.2000, 1.0000],
                    [0.6000, 1.0000],
                    [1.0000, 1.0000]
                ]
            ]
        ],
        dtype=np.float32,
    )
    # Y shape, [N, C, H_out, W_out] - [1, 1, 6, 6]
    Y = np.array(
        [
            [
                [
                    [0.0000, 0.1500, 0.5500, 0.9500, 1.3500, 0.7500],
                    [0.6000, 1.5000, 2.3000, 3.1000, 3.9000, 2.1000],
                    [2.2000, 4.7000, 5.5000, 6.3000, 7.1000, 3.7000],
                    [3.8000, 7.9000, 8.7000, 9.5000, 10.3000, 5.3000],
                    [5.4000, 11.1000, 11.9000, 12.7000, 13.5000, 6.9000],
                    [3.0000, 6.1500, 6.5500, 6.9500, 7.3500, 3.7500]
                ]
            ]
        ],
        dtype=np.float32,
    )
    expect(node, inputs=[X, Grid], outputs=[Y],
           name='test_gridsample')

**gridsample_paddingmode**

::

    # X shape, [N, C, H, W] - [1, 1, 3, 2]
    X = np.array(
        [
            [
                [
                    [0., 1.],
                    [2., 3.],
                    [4., 5.]
                ]
            ]
        ],
        dtype=np.float32,
    )
    # Grid shape, [N, H_out, W_out, 2] - [1, 2, 4, 2]
    Grid = np.array(
        [
            [
                [
                    [-10.0000, -10.0000],
                    [-5.0000, -5.0000],
                    [-0.2000, -0.2000],
                    [10.0000, 10.0000]
                ],

                [
                    [10.0000, 10.0000],
                    [-0.2000, -0.2000],
                    [5.0000, 5.0000],
                    [10.0000, 10.0000]
                ]
            ]
        ],
        dtype=np.float32,
    )

    # setting padding_mode = 'zeros'
    node = onnx.helper.make_node(
        'GridSample',
        inputs=['X', 'Grid'],
        outputs=['Y'],
        padding_mode='zeros',
    )
    # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
    Y_zeros = np.array(
        [
            [
                [
                    [0.0000, 0.0000, 1.7000, 0.0000],
                    [0.0000, 1.7000, 0.0000, 0.0000]
                ]
            ]
        ],
        dtype=np.float32,
    )

    expect(node, inputs=[X, Grid], outputs=[Y_zeros],
           name='test_gridsample_zeros_padding')

    # setting padding_mode = 'border'
    node = onnx.helper.make_node(
        'GridSample',
        inputs=['X', 'Grid'],
        outputs=['Y'],
        padding_mode='border',
    )
    # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
    Y_border = np.array(
        [
            [
                [
                    [0.0000, 0.0000, 1.7000, 5.0000],
                    [5.0000, 1.7000, 5.0000, 5.0000]
                ]
            ]
        ],
        dtype=np.float32,
    )

    expect(node, inputs=[X, Grid], outputs=[Y_border],
           name='test_gridsample_border_padding')

    # setting padding_mode = 'reflection'
    node = onnx.helper.make_node(
        'GridSample',
        inputs=['X', 'Grid'],
        outputs=['Y'],
        padding_mode='reflection',
    )
    # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
    Y_reflection = np.array(
        [
            [
                [
                    [2.5000, 0.0000, 1.7000, 2.5000],
                    [2.5000, 1.7000, 5.0000, 2.5000]
                ]
            ]
        ],
        dtype=np.float32,
    )

    expect(node, inputs=[X, Grid], outputs=[Y_reflection],
           name='test_gridsample_reflection_padding')

**gridsample_mode_aligncorners**

::

        # X shape, [N, C, H, W] - [1, 1, 3, 2]
        X = np.array(
            [
                [
                    [
                        [0., 1.],
                        [2., 3.],
                        [4., 5.]
                    ]
                ]
            ],
            dtype=np.float32,
        )
        # Grid shape, [N, H_out, W_out, 2] - [1, 2, 4, 2]
        Grid = np.array(
            [
                [
                    [
                        [-1.0000, -1.0000],
                        [-0.5000, -0.5000],
                        [-0.2000, -0.2000],
                        [0.0000, 0.0000]
                    ],

                    [
                        [0.0000, 0.0000],
                        [-0.2000, -0.2000],
                        [0.5000, 0.5000],
                        [1.0000, 1.0000]
                    ]
                ]
            ],
            dtype=np.float32,
        )

        # setting mode = 'bilinear', default align_corners = 0
        node = onnx.helper.make_node(
            'GridSample',
            inputs=['X', 'Grid'],
            outputs=['Y'],
            mode='bilinear',
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_bilinear = np.array(
            [
                [
                    [
                        [0.0000, 0.5000, 1.7000, 2.5000],
                        [2.5000, 1.7000, 4.5000, 1.2500]
                    ]
                ]
            ],
            dtype=np.float32,
        )

        expect(node, inputs=[X, Grid], outputs=[Y_bilinear],
               name='test_gridsample_bilinear')

        # setting mode = 'bilinear', align_corners = 1
        node = onnx.helper.make_node(
            'GridSample',
            inputs=['X', 'Grid'],
            outputs=['Y'],
            mode='bilinear',
            align_corners=1,
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_align_corners = np.array(
            [
                [
                    [
                        [0.0000, 1.2500, 2.0000, 2.5000],
                        [2.5000, 2.0000, 3.7500, 5.0000]
                    ]
                ]
            ],
            dtype=np.float32,
        )

        expect(node, inputs=[X, Grid], outputs=[Y_align_corners],
               name='test_gridsample_aligncorners_true')

        # setting mode = 'nearest'
        node = onnx.helper.make_node(
            'GridSample',
            inputs=['X', 'Grid'],
            outputs=['Y'],
            mode='nearest',
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_nearest = np.array(
            [
                [
                    [
                        [0., 0., 2., 2.],
                        [2., 2., 5., 0.]
                    ]
                ]
            ],
            dtype=np.float32,
        )

        expect(node, inputs=[X, Grid], outputs=[Y_nearest],
               name='test_gridsample_nearest')

        # setting mode = 'bicubic'
        node = onnx.helper.make_node(
            'GridSample',
            inputs=['X', 'Grid'],
            outputs=['Y'],
            mode='bicubic',
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_bicubic = np.array(
            [
                [
                    [
                        [-0.1406, 0.3828, 1.7556, 2.9688],
                        [2.9688, 1.7556, 5.1445, 1.3906]
                    ]
                ]
            ],
            dtype=np.float32,
        )

        expect(node, inputs=[X, Grid], outputs=[Y_bicubic],
               name='test_gridsample_bicubic')

    '''
    For someone who want to test by script. Comment it cause github ONNX CI
    do not have the torch python package.
