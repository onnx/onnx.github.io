
.. _l-onnx-doc-Slice:

=====
Slice
=====

.. contents::
    :local:


.. _l-onnx-op-slice-13:

Slice - 13
==========

**Version**

* **name**: `Slice (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Slice>`_
* **domain**: **main**
* **since_version**: **13**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 13**.

**Summary**

Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://numpy.org/doc/stable/user/basics.indexing.html?highlight=slice#slicing-and-striding

Slice uses the `starts`, `ends`, `axes` and `steps` inputs to select a sub-tensor
of its input `data` tensor.

An effective `start[i]`, `end[i]`, and `step[i]` must be computed for each `i`
in `[0, ... r-1]` where `r = rank(input)` as follows:

If `axes` are omitted, they are set to `[0, ..., r-1]`.
If `steps` are omitted, they are set to `[1, ..., 1]` of length `len(starts)`

The effective values are initialized as `start[i] = 0`, `end[i] = dims[i]` where
`dims` are the dimensions of `input` and `step[i] = `1.

All negative elements of `axes` are made non-negatve by adding `r` to them, where
`r =rank(input)`.

All negative values in `starts[i]` and `ends[i]` have `dims[axes[i]]` added to them,
where `dims` are the dimensions of `input`. Then `start[axes[i]]` is the adjusted
`starts[i]` is clamped into the range `[0, dims[axes[i]]]` for positive stepping
and `[0, dims[axes[i]]-1]` for negative stepping.

The clamping for the adjusted `ends[i]` depends on the sign of `steps[i]` and must
accommodate copying 0 through `dims[axes[i]]` elements, so for positive stepping
`end[axes[i]]` is clamped to `[0, dims[axes[i]]]`, while for negative stepping it
is clamped to `[-1, dims[axes[i]]-1]`.

Finally, `step[axes[i]] = steps[i]`.

For slicing to the end of a dimension with unknown size, it is recommended to pass
in `INT_MAX` when slicing forward and 'INT_MIN' when slicing backward.

Example 1:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  axes = [0, 1]
  starts = [1, 0]
  ends = [2, 3]
  steps = [1, 2]
  result = [
      [5, 7],
  ]
Example 2:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 1000]
  result = [
      [2, 3, 4],
  ]

**Inputs**

Between 3 and 5 inputs.

* **data** (heterogeneous) - **T**:
  Tensor of data to extract slices from.
* **starts** (heterogeneous) - **Tind**:
  1-D tensor of starting indices of corresponding axis in `axes`
* **ends** (heterogeneous) - **Tind**:
  1-D tensor of ending indices (exclusive) of corresponding axis in
  `axes`
* **axes** (optional, heterogeneous) - **Tind**:
  1-D tensor of axes that `starts` and `ends` apply to. Negative value
  means counting dimensions from the back. Accepted range is [-r, r-1]
  where r = rank(data). Behavior is undefined if an axis is repeated.
* **steps** (optional, heterogeneous) - **Tind**:
  1-D tensor of slice step of corresponding axis in `axes`. Negative
  value means slicing backward. 'steps' cannot be 0. Defaults to 1s.

**Outputs**

* **output** (heterogeneous) - **T**:
  Sliced data tensor.

**Type Constraints**

* **T** in (
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
  Constrain input and output types to all tensor types.
* **Tind** in (
  tensor(int32),
  tensor(int64)
  ):
  Constrain indices to integer types

**Examples**

**slice**

::

    node = onnx.helper.make_node(
        'Slice',
        inputs=['x', 'starts', 'ends', 'axes', 'steps'],
        outputs=['y'],
    )

    x = np.random.randn(20, 10, 5).astype(np.float32)
    y = x[0:3, 0:10]
    starts = np.array([0, 0], dtype=np.int64)
    ends = np.array([3, 10], dtype=np.int64)
    axes = np.array([0, 1], dtype=np.int64)
    steps = np.array([1, 1], dtype=np.int64)

    expect(node, inputs=[x, starts, ends, axes, steps], outputs=[y],
           name='test_slice')

**slice_neg**

::

    node = onnx.helper.make_node(
        'Slice',
        inputs=['x', 'starts', 'ends', 'axes', 'steps'],
        outputs=['y'],
    )

    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([0], dtype=np.int64)
    ends = np.array([-1], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    y = x[:, 0:-1]

    expect(node, inputs=[x, starts, ends, axes, steps], outputs=[y],
           name='test_slice_neg')

**slice_start_out_of_bounds**

::

    node = onnx.helper.make_node(
        'Slice',
        inputs=['x', 'starts', 'ends', 'axes', 'steps'],
        outputs=['y'],
    )

    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([1000], dtype=np.int64)
    ends = np.array([1000], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    y = x[:, 1000:1000]

    expect(node, inputs=[x, starts, ends, axes, steps], outputs=[y],
           name='test_slice_start_out_of_bounds')

**slice_end_out_of_bounds**

::

    node = onnx.helper.make_node(
        'Slice',
        inputs=['x', 'starts', 'ends', 'axes', 'steps'],
        outputs=['y'],
    )

    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([1], dtype=np.int64)
    ends = np.array([1000], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    y = x[:, 1:1000]

    expect(node, inputs=[x, starts, ends, axes, steps], outputs=[y],
           name='test_slice_end_out_of_bounds')

**slice_default_axes**

::

    node = onnx.helper.make_node(
        'Slice',
        inputs=['x', 'starts', 'ends'],
        outputs=['y'],
    )

    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([0, 0, 3], dtype=np.int64)
    ends = np.array([20, 10, 4], dtype=np.int64)
    y = x[:, :, 3:4]

    expect(node, inputs=[x, starts, ends], outputs=[y],
           name='test_slice_default_axes')

**slice_default_steps**

::

    node = onnx.helper.make_node(
        'Slice',
        inputs=['x', 'starts', 'ends', 'axes'],
        outputs=['y'],
    )

    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([0, 0, 3], dtype=np.int64)
    ends = np.array([20, 10, 4], dtype=np.int64)
    axes = np.array([0, 1, 2], dtype=np.int64)
    y = x[:, :, 3:4]

    expect(node, inputs=[x, starts, ends, axes], outputs=[y],
           name='test_slice_default_steps')

**slice_neg_steps**

::

    node = onnx.helper.make_node(
        'Slice',
        inputs=['x', 'starts', 'ends', 'axes', 'steps'],
        outputs=['y'],
    )

    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([20, 10, 4], dtype=np.int64)
    ends = np.array([0, 0, 1], dtype=np.int64)
    axes = np.array([0, 1, 2], dtype=np.int64)
    steps = np.array([-1, -3, -2]).astype(np.int64)
    y = x[20:0:-1, 10:0:-3, 4:1:-2]

    expect(node, inputs=[x, starts, ends, axes, steps], outputs=[y],
           name='test_slice_neg_steps')

**slice_negative_axes**

::

    node = onnx.helper.make_node(
        'Slice',
        inputs=['x', 'starts', 'ends', 'axes'],
        outputs=['y'],
    )

    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([0, 0, 3], dtype=np.int64)
    ends = np.array([20, 10, 4], dtype=np.int64)
    axes = np.array([0, -2, -1], dtype=np.int64)
    y = x[:, :, 3:4]

    expect(node, inputs=[x, starts, ends, axes], outputs=[y],
           name='test_slice_negative_axes')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Produces a slice of the input tensor along multiple axes. Similar to numpy:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Produces a slice of the input tensor along multiple axes. Similar to numpy:</code></td></tr>
    <tr style="1px solid black;"><td><code>1</code></td><td><code>1</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">https://<span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">.</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">i</span>py.org/doc/<span style="color:#BA4A00;">n</span>u<span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">y</span><span style="color:#BA4A00;">/</span><span style="color:#BA4A00;">r</span>e<span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">e</span>r<span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">e</span>/a<span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">y</span>s.indexing.html</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>https://<span style="color:#196F3D;">n</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span>py.org/doc/<span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">/</span>u<span style="color:#196F3D;">s</span>er/<span style="color:#196F3D;">b</span>as<span style="color:#196F3D;">i</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">s</span>.indexing.html<span style="color:#196F3D;">?</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">=</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">#</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">2</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td><code>2</code></td><td><code>3</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">Slice<span style="color:#BA4A00;">s</span> uses starts, ends, axes and steps inputs to s<span style="color:#BA4A00;">p</span>ec<span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">y</span><span style="color:#BA4A00;"> </span>t<span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span> st<span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;"> </span>en<span style="color:#BA4A00;">d</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>Slice uses <span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span>starts, ends, axes and steps inputs to se<span style="color:#196F3D;">l</span><span style="color:#196F3D;">e</span>ct <span style="color:#196F3D;">a</span><span style="color:#196F3D;"> </span>s<span style="color:#196F3D;">u</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">-</span>ten<span style="color:#196F3D;">s</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span></code></td></tr>
    <tr style="1px solid black;"><td><code>3</code></td><td><code>4</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">i</span>o<span style="color:#BA4A00;">n</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;"> </span>f<span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span> <span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">x</span>is in<span style="color:#BA4A00;"> </span>t<span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span> <span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span>t<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span>a<span style="color:#BA4A00;">x</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">,</span> <span style="color:#BA4A00;">i</span>t<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">s</span>es<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">f</span>or<span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">o</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>of i<span style="color:#196F3D;">t</span>s in<span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span>t <span style="color:#196F3D;">d</span><span style="color:#196F3D;">a</span>ta te<span style="color:#196F3D;">n</span>sor<span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">5</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">6</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">An effective start[i], end[i], and step[i] must be computed for each i</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">7</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">in [0, ... r-1] where r = rank(input) as follows:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">8</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">9</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">If axes are omitted, they are set to [0, ..., r-1].</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">10</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">If steps are omitted, they are set to [1, ..., 1] of length len(starts)</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">11</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">12</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">The effective values are initialized as start[i] = 0, end[i] = dims[i] where</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">13</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">dims are the dimensions of input and step[i] = 1.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">14</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">15</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">All negative elements of axes are made non-negatve by adding r to them, where</code></td></tr>
    <tr style="1px solid black;"><td><code>4</code></td><td><code>16</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">e</span> <span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span>input<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span>.<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">I</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">v</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">v</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">y</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">r</span> <span style="color:#196F3D;">=</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">k</span><span style="color:#196F3D;">(</span>input<span style="color:#196F3D;">)</span>.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">17</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">18</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">All negative values in starts[i] and ends[i] have dims[axes[i]] added to them,</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">19</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">where dims are the dimensions of input. Then start[axes[i]] is the adjusted</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">20</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">starts[i] is clamped into the range [0, dims[axes[i]]] for positive stepping</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">21</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">and [0, dims[axes[i]]-1] for negative stepping.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">22</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">23</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">The clamping for the adjusted ends[i] depends on the sign of steps[i] and must</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">24</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">accommodate copying 0 through dims[axes[i]] elements, so for positive stepping</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">25</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">end[axes[i]] is clamped to [0, dims[axes[i]]], while for negative stepping it</code></td></tr>
    <tr style="1px solid black;"><td><code>5</code></td><td><code>26</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">s<span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">t</span> <span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">i</span>c<span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">e</span>p<span style="color:#BA4A00;">r</span>e<span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">s</span> <span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span>t<span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">f</span>o<span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">e</span> <span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span> <span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span>d<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span>a<span style="color:#BA4A00;">t</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">i</span>s c<span style="color:#196F3D;">l</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">m</span>pe<span style="color:#196F3D;">d</span> to <span style="color:#196F3D;">[</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">,</span> d<span style="color:#196F3D;">i</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">[</span>a<span style="color:#196F3D;">x</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">[</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">]</span><span style="color:#196F3D;">]</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">]</span><span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">27</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td><code>6</code></td><td><code>28</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">d</span>i<span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">e</span>ns<span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">.</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">I</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span>t<span style="color:#BA4A00;">h</span>e<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">v</span>a<span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">u</span>e<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">a</span>s<span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">d</span> <span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">o</span> st<span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;"> </span>e<span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;"> </span>i<span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">(</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">F</span>in<span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span>ste<span style="color:#196F3D;">p</span><span style="color:#196F3D;">[</span>a<span style="color:#196F3D;">x</span>es<span style="color:#196F3D;">[</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">]</span><span style="color:#196F3D;">]</span> <span style="color:#196F3D;">=</span> ste<span style="color:#196F3D;">p</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">[</span>i<span style="color:#196F3D;">]</span><span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">7</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">number of elements in this dimension), it represents n. For slicing to the</code></td><td></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">29</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td><code>8</code></td><td><code>30</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">end of a dimension with unknown size, it is recommended to pass<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">I</span><span style="color:#BA4A00;">N</span><span style="color:#BA4A00;">T</span><span style="color:#BA4A00;">_</span><span style="color:#BA4A00;">M</span><span style="color:#BA4A00;">A</span><span style="color:#BA4A00;">X</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">F</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span>e<span style="color:#196F3D;"> </span><span style="color:#196F3D;">e</span>nd of a dimension with unknown size, it is recommended to pass</code></td></tr>
    <tr style="1px solid black;"><td><code>9</code></td><td><code>31</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">when s<span style="color:#BA4A00;">c</span>licing forward and 'INT_MIN' when slicing backward.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">I</span><span style="color:#196F3D;">N</span><span style="color:#196F3D;">T</span><span style="color:#196F3D;">_</span><span style="color:#196F3D;">M</span><span style="color:#196F3D;">A</span><span style="color:#196F3D;">X</span><span style="color:#196F3D;"> </span>when slicing forward and 'INT_MIN' when slicing backward.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">10</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">If a negative value is passed for step, it represents slicing backward.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">11</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">However step value cannot be 0.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">12</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">If axes are omitted, they are set to [0, ..., ndim-1].</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">13</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">If steps are omitted, they are set to [1, ..., 1] of length len(starts)</code></td><td></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">32</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 1:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 1:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data = [</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data = [</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [1, 2, 3, 4],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [1, 2, 3, 4],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [5, 6, 7, 8],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [5, 6, 7, 8],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  axes = [0, 1]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  axes = [0, 1]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  starts = [1, 0]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  starts = [1, 0]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ends = [2, 3]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ends = [2, 3]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  steps = [1, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  steps = [1, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  result = [</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  result = [</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [5, 7],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [5, 7],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 2:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 2:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data = [</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data = [</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [1, 2, 3, 4],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [1, 2, 3, 4],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [5, 6, 7, 8],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [5, 6, 7, 8],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  starts = [0, 1]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  starts = [0, 1]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ends = [-1, 1000]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ends = [-1, 1000]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  result = [</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  result = [</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [2, 3, 4],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [2, 3, 4],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 3 and 5 inputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 3 and 5 inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of data to extract slices from.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of data to extract slices from.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **starts** (heterogeneous) - **Tind**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **starts** (heterogeneous) - **Tind**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  1-D tensor of starting indices of corresponding axis in axes</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  1-D tensor of starting indices of corresponding axis in axes</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **ends** (heterogeneous) - **Tind**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **ends** (heterogeneous) - **Tind**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  1-D tensor of ending indices (exclusive) of corresponding axis in</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  1-D tensor of ending indices (exclusive) of corresponding axis in</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  axes</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  axes</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **axes** (optional, heterogeneous) - **Tind**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **axes** (optional, heterogeneous) - **Tind**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  1-D tensor of axes that starts and ends apply to. Negative value</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  1-D tensor of axes that starts and ends apply to. Negative value</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  means counting dimensions from the back. Accepted range is [-r, r-1]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  means counting dimensions from the back. Accepted range is [-r, r-1]</code></td></tr>
    <tr style="1px solid black;"><td><code>51</code></td><td><code>70</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  where r = rank(data).</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  where r = rank(data).<span style="color:#196F3D;"> </span><span style="color:#196F3D;">B</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **steps** (optional, heterogeneous) - **Tind**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **steps** (optional, heterogeneous) - **Tind**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  1-D tensor of slice step of corresponding axis in axes. Negative</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  1-D tensor of slice step of corresponding axis in axes. Negative</code></td></tr>
    <tr style="1px solid black;"><td><code>54</code></td><td><code>73</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  value means slicing backward. 'steps' cannot be 0. Defaults to 1.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  value means slicing backward. 'steps' cannot be 0. Defaults to 1<span style="color:#196F3D;">s</span>.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Sliced data tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Sliced data tensor.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">83</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(bfloat16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to all tensor types.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to all tensor types.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Tind** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Tind** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain indices to integer types</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain indices to integer types</code></td></tr>
    </table>

.. _l-onnx-op-slice-11:

Slice - 11
==========

**Version**

* **name**: `Slice (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Slice>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
Slices uses `starts`, `ends`, `axes` and `steps` inputs to specify the start and end
dimension and step for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represents number of elements before the end of that
dimension. If the value passed to start or end is larger than the `n` (the
number of elements in this dimension), it represents `n`. For slicing to the
end of a dimension with unknown size, it is recommended to pass in `INT_MAX`
when sclicing forward and 'INT_MIN' when slicing backward.
If a negative value is passed for step, it represents slicing backward.
However step value cannot be 0.
If `axes` are omitted, they are set to `[0, ..., ndim-1]`.
If `steps` are omitted, they are set to `[1, ..., 1]` of length `len(starts)`
Example 1:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  axes = [0, 1]
  starts = [1, 0]
  ends = [2, 3]
  steps = [1, 2]
  result = [
      [5, 7],
  ]
Example 2:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 1000]
  result = [
      [2, 3, 4],
  ]

**Inputs**

Between 3 and 5 inputs.

* **data** (heterogeneous) - **T**:
  Tensor of data to extract slices from.
* **starts** (heterogeneous) - **Tind**:
  1-D tensor of starting indices of corresponding axis in `axes`
* **ends** (heterogeneous) - **Tind**:
  1-D tensor of ending indices (exclusive) of corresponding axis in
  `axes`
* **axes** (optional, heterogeneous) - **Tind**:
  1-D tensor of axes that `starts` and `ends` apply to. Negative value
  means counting dimensions from the back. Accepted range is [-r, r-1]
  where r = rank(data).
* **steps** (optional, heterogeneous) - **Tind**:
  1-D tensor of slice step of corresponding axis in `axes`. Negative
  value means slicing backward. 'steps' cannot be 0. Defaults to 1.

**Outputs**

* **output** (heterogeneous) - **T**:
  Sliced data tensor.

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
  Constrain input and output types to all tensor types.
* **Tind** in (
  tensor(int32),
  tensor(int64)
  ):
  Constrain indices to integer types

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Produces a slice of the input tensor along multiple axes. Similar to numpy:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Produces a slice of the input tensor along multiple axes. Similar to numpy:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Slices uses starts, ends, axes and steps inputs to specify the start and end</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Slices uses starts, ends, axes and steps inputs to specify the start and end</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">dimension and step for each axis in the list of axes, it uses this information to</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">dimension and step for each axis in the list of axes, it uses this information to</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">slice the input data tensor. If a negative value is passed for any of the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">slice the input data tensor. If a negative value is passed for any of the</code></td></tr>
    <tr style="1px solid black;"><td><code>5</code></td><td><code>5</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">start or end indices, it represent number of elements before the end of that</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>start or end indices, it represent<span style="color:#196F3D;">s</span> number of elements before the end of that</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">dimension. If the value passed to start or end is larger than the n (the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">dimension. If the value passed to start or end is larger than the n (the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">number of elements in this dimension), it represents n. For slicing to the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">number of elements in this dimension), it represents n. For slicing to the</code></td></tr>
    <tr style="1px solid black;"><td><code>8</code></td><td><code>8</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">end of a dimension with unknown size, it is recommended to pass in INT_MAX<span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>end of a dimension with unknown size, it is recommended to pass in INT_MAX</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">9</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">when sclicing forward and 'INT_MIN' when slicing backward.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If a negative value is passed for step, it represents slicing backward.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If a negative value is passed for step, it represents slicing backward.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">11</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">However step value cannot be 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If axes are omitted, they are set to [0, ..., ndim-1].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If axes are omitted, they are set to [0, ..., ndim-1].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If steps are omitted, they are set to [1, ..., 1] of length len(starts)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If steps are omitted, they are set to [1, ..., 1] of length len(starts)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 1:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 1:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data = [</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data = [</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [1, 2, 3, 4],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [1, 2, 3, 4],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [5, 6, 7, 8],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [5, 6, 7, 8],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  axes = [0, 1]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  axes = [0, 1]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  starts = [1, 0]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  starts = [1, 0]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ends = [2, 3]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ends = [2, 3]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  steps = [1, 2]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  steps = [1, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  result = [</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  result = [</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [5, 7],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [5, 7],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 2:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 2:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data = [</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data = [</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [1, 2, 3, 4],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [1, 2, 3, 4],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [5, 6, 7, 8],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [5, 6, 7, 8],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  starts = [0, 1]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  starts = [0, 1]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ends = [-1, 1000]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ends = [-1, 1000]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  result = [</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  result = [</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [2, 3, 4],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [2, 3, 4],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 3 and 5 inputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 3 and 5 inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of data to extract slices from.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of data to extract slices from.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **starts** (heterogeneous) - **Tind**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **starts** (heterogeneous) - **Tind**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  1-D tensor of starting indices of corresponding axis in axes</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  1-D tensor of starting indices of corresponding axis in axes</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **ends** (heterogeneous) - **Tind**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **ends** (heterogeneous) - **Tind**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  1-D tensor of ending indices (exclusive) of corresponding axis in</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  1-D tensor of ending indices (exclusive) of corresponding axis in</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  axes</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  axes</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **axes** (optional, heterogeneous) - **Tind**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **axes** (optional, heterogeneous) - **Tind**:</code></td></tr>
    <tr style="1px solid black;"><td><code>47</code></td><td><code>49</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  1-D tensor of axes that starts and ends apply to.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  1-D tensor of axes that starts and ends apply to.<span style="color:#196F3D;"> </span><span style="color:#196F3D;">N</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">e</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">50</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  means counting dimensions from the back. Accepted range is [-r, r-1]</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">51</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  where r = rank(data).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **steps** (optional, heterogeneous) - **Tind**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **steps** (optional, heterogeneous) - **Tind**:</code></td></tr>
    <tr style="1px solid black;"><td><code>49</code></td><td><code>53</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  1-D tensor of slice step of corresponding axis in axes. <span style="color:#BA4A00;">D</span>e<span style="color:#BA4A00;">f</span>a<span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">l</span>t<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">o</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  1-D tensor of slice step of corresponding axis in axes. <span style="color:#196F3D;">N</span>e<span style="color:#196F3D;">g</span>at<span style="color:#196F3D;">i</span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">e</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">50</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  1.</code></td><td></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">54</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  value means slicing backward. 'steps' cannot be 0. Defaults to 1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Sliced data tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Sliced data tensor.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to all tensor types.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to all tensor types.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Tind** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Tind** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain indices to integer types</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain indices to integer types</code></td></tr>
    </table>

.. _l-onnx-op-slice-10:

Slice - 10
==========

**Version**

* **name**: `Slice (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Slice>`_
* **domain**: **main**
* **since_version**: **10**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 10**.

**Summary**

Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
Slices uses `starts`, `ends`, `axes` and `steps` inputs to specify the start and end
dimension and step for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represent number of elements before the end of that
dimension. If the value passed to start or end is larger than the `n` (the
number of elements in this dimension), it represents `n`. For slicing to the
end of a dimension with unknown size, it is recommended to pass in `INT_MAX`.
If a negative value is passed for step, it represents slicing backward.
If `axes` are omitted, they are set to `[0, ..., ndim-1]`.
If `steps` are omitted, they are set to `[1, ..., 1]` of length `len(starts)`
Example 1:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  axes = [0, 1]
  starts = [1, 0]
  ends = [2, 3]
  steps = [1, 2]
  result = [
      [5, 7],
  ]
Example 2:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 1000]
  result = [
      [2, 3, 4],
  ]

**Inputs**

Between 3 and 5 inputs.

* **data** (heterogeneous) - **T**:
  Tensor of data to extract slices from.
* **starts** (heterogeneous) - **Tind**:
  1-D tensor of starting indices of corresponding axis in `axes`
* **ends** (heterogeneous) - **Tind**:
  1-D tensor of ending indices (exclusive) of corresponding axis in
  `axes`
* **axes** (optional, heterogeneous) - **Tind**:
  1-D tensor of axes that `starts` and `ends` apply to.
* **steps** (optional, heterogeneous) - **Tind**:
  1-D tensor of slice step of corresponding axis in `axes`. Default to
  1.

**Outputs**

* **output** (heterogeneous) - **T**:
  Sliced data tensor.

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
  Constrain input and output types to all tensor types.
* **Tind** in (
  tensor(int32),
  tensor(int64)
  ):
  Constrain indices to integer types

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Produces a slice of the input tensor along multiple axes. Similar to numpy:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Produces a slice of the input tensor along multiple axes. Similar to numpy:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html</code></td></tr>
    <tr style="1px solid black;"><td><code>2</code></td><td><code>2</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">Slices uses <span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">x</span><span style="color:#BA4A00;">e</span>s<span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span>tarts <span style="color:#BA4A00;">a</span>nd ends <span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">r</span>i<span style="color:#BA4A00;">b</span>ut<span style="color:#BA4A00;">e</span>s to specify the start and end</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>Slices uses starts<span style="color:#196F3D;">,</span> <span style="color:#196F3D;">e</span>nd<span style="color:#196F3D;">s</span><span style="color:#196F3D;">,</span> <span style="color:#196F3D;">a</span><span style="color:#196F3D;">x</span>e<span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span>nd<span style="color:#196F3D;"> </span>s<span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">s</span> i<span style="color:#196F3D;">n</span><span style="color:#196F3D;">p</span>uts to specify the start and end</code></td></tr>
    <tr style="1px solid black;"><td><code>3</code></td><td><code>3</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">dimension for each axis in the list of axes, it uses this information to</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>dimension <span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;"> </span>for each axis in the list of axes, it uses this information to</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">slice the input data tensor. If a negative value is passed for any of the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">slice the input data tensor. If a negative value is passed for any of the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">start or end indices, it represent number of elements before the end of that</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">start or end indices, it represent number of elements before the end of that</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">dimension. If the value passed to start or end is larger than the n (the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">dimension. If the value passed to start or end is larger than the n (the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">number of elements in this dimension), it represents n. For slicing to the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">number of elements in this dimension), it represents n. For slicing to the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">end of a dimension with unknown size, it is recommended to pass in INT_MAX.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">end of a dimension with unknown size, it is recommended to pass in INT_MAX.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">9</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">If a negative value is passed for step, it represents slicing backward.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If axes are omitted, they are set to [0, ..., ndim-1].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If axes are omitted, they are set to [0, ..., ndim-1].</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">11</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">If steps are omitted, they are set to [1, ..., 1] of length len(starts)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 1:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 1:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data = [</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data = [</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [1, 2, 3, 4],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [1, 2, 3, 4],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [5, 6, 7, 8],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [5, 6, 7, 8],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  axes = [0, 1]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  axes = [0, 1]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  starts = [1, 0]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  starts = [1, 0]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ends = [2, 3]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ends = [2, 3]</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">20</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  steps = [1, 2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  result = [</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  result = [</code></td></tr>
    <tr style="1px solid black;"><td><code>19</code></td><td><code>22</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      [5, <span style="color:#BA4A00;">6</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span>7],</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      [5, 7],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 2:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 2:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data = [</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data = [</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [1, 2, 3, 4],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [1, 2, 3, 4],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [5, 6, 7, 8],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [5, 6, 7, 8],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  starts = [0, 1]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  starts = [0, 1]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ends = [-1, 1000]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ends = [-1, 1000]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  result = [</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  result = [</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [2, 3, 4],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      [2, 3, 4],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">32</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">**Attributes**</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">33</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;"></code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">34</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">* **axes**:</code></td><td></td></tr>
    <tr style="1px solid black;"><td><code>35</code></td><td><code>35</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">A</span><span style="color:#BA4A00;">x</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span>n<span style="color:#BA4A00;">d</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span>p<span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">y</span><span style="color:#BA4A00;"> </span>t<span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">.</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">I</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">'</span>s<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">.</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">I</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">t</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">I</span>np<span style="color:#196F3D;">u</span>ts<span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">36</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td><code>36</code></td><td><code>37</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">r</span>e<span style="color:#BA4A00;">s</span>en<span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">,</span> <span style="color:#BA4A00;">w</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">l</span> <span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">e</span>a<span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span>d <span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">s</span> <span style="color:#BA4A00;">[</span><span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">1</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">.</span><span style="color:#BA4A00;">.</span><span style="color:#BA4A00;">.</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">e</span>n<span style="color:#BA4A00;">(</span><span style="color:#BA4A00;">s</span>t<span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">t</span>s<span style="color:#BA4A00;">)</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">-</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">1</span><span style="color:#BA4A00;">]</span>.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">B</span>e<span style="color:#196F3D;">t</span><span style="color:#196F3D;">w</span>e<span style="color:#196F3D;">e</span>n <span style="color:#196F3D;">3</span> a<span style="color:#196F3D;">n</span>d <span style="color:#196F3D;">5</span> <span style="color:#196F3D;">i</span>n<span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span>ts.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">38</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">39</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **data** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">40</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Tensor of data to extract slices from.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">41</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **starts** (heterogeneous) - **Tind**:</code></td></tr>
    <tr style="1px solid black;"><td><code>37</code></td><td><code>42</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">*</span> <span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">*</span>ends<span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">*</span> <span style="color:#BA4A00;">(</span>re<span style="color:#BA4A00;">q</span><span style="color:#BA4A00;">u</span>i<span style="color:#BA4A00;">r</span>e<span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">)</span><span style="color:#BA4A00;">:</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code> <span style="color:#196F3D;"> </span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">D</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span>en<span style="color:#196F3D;">s</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span>d<span style="color:#196F3D;">i</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">e</span>s <span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">o</span>r<span style="color:#196F3D;">r</span>e<span style="color:#196F3D;">s</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span>i<span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">x</span>e<span style="color:#196F3D;">s</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">43</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **ends** (heterogeneous) - **Tind**:</code></td></tr>
    <tr style="1px solid black;"><td><code>38</code></td><td><code>44</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  <span style="color:#BA4A00;">E</span>nding indices (exclusive) of corresponding axis in<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">x</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">s</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">1</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">D</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span>n<span style="color:#196F3D;">s</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span>ding indices (exclusive) of corresponding axis in</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">39</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">* **starts** (required):</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">40</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  Starting indices of corresponding axis in axes</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">41</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;"></code></td><td></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">45</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  axes</code></td></tr>
    <tr style="1px solid black;"><td><code>42</code></td><td><code>46</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">**<span style="color:#BA4A00;">I</span>n<span style="color:#BA4A00;">p</span>u<span style="color:#BA4A00;">t</span>s**</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>*<span style="color:#196F3D;"> </span>*<span style="color:#196F3D;">*</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span>n<span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">o</span>us<span style="color:#196F3D;">)</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">-</span><span style="color:#196F3D;"> </span>**<span style="color:#196F3D;">T</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">:</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">43</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;"></code></td><td></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">47</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  1-D tensor of axes that starts and ends apply to.</code></td></tr>
    <tr style="1px solid black;"><td><code>44</code></td><td><code>48</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">* **<span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">a</span>t<span style="color:#BA4A00;">a</span>** (heterogeneous) - **T**:</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>* **<span style="color:#196F3D;">s</span>t<span style="color:#196F3D;">e</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">s</span>** (<span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span>heterogeneous) - **T<span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span>**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">45</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  Tensor of data to extract slices from.</code></td><td></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">49</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  1-D tensor of slice step of corresponding axis in axes. Default to</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">50</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Sliced data tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Sliced data tensor.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to all tensor types.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to all tensor types.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">77</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **Tind** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">78</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">79</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">80</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  ):</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">81</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Constrain indices to integer types</code></td></tr>
    </table>

.. _l-onnx-op-slice-1:

Slice - 1
=========

**Version**

* **name**: `Slice (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Slice>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1**.

**Summary**

Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
Slices uses `axes`, `starts` and `ends` attributes to specify the start and end
dimension for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represent number of elements before the end of that
dimension. If the value passed to start or end is larger than the `n` (the
number of elements in this dimension), it represents `n`. For slicing to the
end of a dimension with unknown size, it is recommended to pass in `INT_MAX`.
If `axes` are omitted, they are set to `[0, ..., ndim-1]`.
Example 1:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  axes = [0, 1]
  starts = [1, 0]
  ends = [2, 3]
  result = [
      [5, 6, 7],
  ]
Example 2:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 1000]
  result = [
      [2, 3, 4],
  ]

**Attributes**

* **axes**:
  Axes that `starts` and `ends` apply to. It's optional. If not
  present, will be treated as [0, 1, ..., len(`starts`) - 1].
* **ends** (required):
  Ending indices (exclusive) of corresponding axis in axes`
* **starts** (required):
  Starting indices of corresponding axis in `axes`

**Inputs**

* **data** (heterogeneous) - **T**:
  Tensor of data to extract slices from.

**Outputs**

* **output** (heterogeneous) - **T**:
  Sliced data tensor.

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
  Constrain input and output types to all tensor types.
