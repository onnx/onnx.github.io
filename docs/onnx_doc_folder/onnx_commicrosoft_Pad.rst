
.. _l-onnx-doccom.microsoft-Pad:

===================
com.microsoft - Pad
===================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-pad-1:

Pad - 1 (com.microsoft)
=======================

**Version**

* **name**: `Pad (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.Pad>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Given `data` tensor, pads, mode, and value.
Example:
Insert 0 pads to the beginning of the second dimension.
data = [
        [1.0, 1.2],
        [2.3, 3.4],
        [4.5, 5.7],
        ]
pads = [0, 2, 0, 0]
output = [
        [
        [0.0, 0.0, 1.0, 1.2],
        [0.0, 0.0, 2.3, 3.4],
        [0.0, 0.0, 4.5, 5.7],
        ],
        ]

**Attributes**

* **mode**:
  Three modes: `constant`(default) - pads with a given constant value,
  `reflect` - pads with the reflection of the vector mirrored on the
  first and last values of the vector along each axis, `edge` - pads
  with the edge values of array Default value is ``?``.

**Inputs**

Between 2 and 3 inputs.

* **data** (heterogeneous) - **T**:
  Input tensor.
* **pads** (heterogeneous) - **tensor(int64)**:
  Tensor of integers indicating the number of padding elements to add
  or remove (if negative) at the beginning and end of each axis. For
  2D input tensor, it is the number of pixels. `pads` should be a 1D
  tensor of shape [2 * input_rank] or a 2D tensor of shape [1, 2 *
  input_rank]. `pads` format (1D example) should be as follow
  [x1_begin, x2_begin,...,x1_end, x2_end,...], where xi_begin is the
  number of pixels added at the beginning of axis `i` and xi_end, the
  number of pixels added at the end of axis `i`.
* **value** (optional, heterogeneous) - **T**:
  (Optional) A scalar or rank 1 tensor containing a single value to be
  filled if the mode chosen is `constant` (by default it is 0.0).

**Outputs**

* **output** (heterogeneous) - **T**:
  Tensor after padding.

**Examples**

**constant_pad**

::

    node = onnx.helper.make_node(
        'Pad',
        inputs=['x', 'pads', 'value'],
        outputs=['y'],
        mode='constant'
    )
    x = np.random.randn(1, 3, 4, 5).astype(np.float32)
    pads = np.array([0, 0, 1, 3, 0, 0, 2, 4]).astype(np.int64)  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
    value = np.float32(1.2)
    y = pad_impl(
        x,
        pads,
        'constant',
        1.2
    )

    expect(node, inputs=[x, pads, value], outputs=[y],
           name='test_constant_pad')

**reflection_and_edge_pad**

::

    for mode in ['edge', 'reflect']:
        node = onnx.helper.make_node(
            'Pad',
            inputs=['x', 'pads'],
            outputs=['y'],
            mode=mode
        )
        x = np.random.randn(1, 3, 4, 5).astype(np.int32)
        pads = np.array([0, 0, 1, 1, 0, 0, 1, 1]).astype(np.int64)  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        y = pad_impl(
            x,
            pads,
            mode
        )

        expect(node, inputs=[x, pads], outputs=[y],
               name=f'test_{mode}_pad')
