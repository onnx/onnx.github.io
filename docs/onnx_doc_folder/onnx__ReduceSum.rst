
.. _l-onnx-doc-ReduceSum:

=========
ReduceSum
=========

.. contents::
    :local:


.. _l-onnx-op-reducesum-13:

ReduceSum - 13
==============

**Version**

* **name**: `ReduceSum (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSum>`_
* **domain**: **main**
* **since_version**: **13**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 13**.

**Summary**

Computes the sum of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equals 0, then
the resulting tensor has the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.

**Attributes**

* **keepdims**:
  Keep the reduced dimension or not, default 1 means keep reduced
  dimension. Default value is ``1``.
* **noop_with_empty_axes**:
  Defines behaviour if 'axes' is empty. Default behaviour with 'false'
  is to reduce all axes. When axes is empty and this attribute is set
  to true, input tensor will not be reduced,and the output tensor
  would be equivalent to input tensor. Default value is ``0``.

**Inputs**

Between 1 and 2 inputs.

* **data** (heterogeneous) - **T**:
  An input tensor.
* **axes** (optional, heterogeneous) - **tensor(int64)**:
  Optional input list of integers, along which to reduce. The default
  is to reduce over all the dimensions of the input tensor if
  'noop_with_empty_axes' is false, else act as an Identity op when
  'noop_with_empty_axes' is true. Accepted range is [-r, r-1] where r
  = rank(data).

**Outputs**

* **reduced** (heterogeneous) - **T**:
  Reduced output tensor.

**Type Constraints**

* **T** in (
  tensor(bfloat16),
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int32),
  tensor(int64),
  tensor(uint32),
  tensor(uint64)
  ):
  Constrain input and output types to high-precision numeric tensors.

**Examples**

**do_not_keepdims**

::

    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = 0

    node = onnx.helper.make_node(
        'ReduceSum',
        inputs=['data', 'axes'],
        outputs=['reduced'],
        keepdims=keepdims)

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
    reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)
    #print(reduced)
    #[[4., 6.]
    # [12., 14.]
    # [20., 22.]]

    expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_do_not_keepdims_example')

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)

    expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_do_not_keepdims_random')

**keepdims**

::

    shape = [3, 2, 2]
    axes = np.array([1], dtype=np.int64)
    keepdims = 1

    node = onnx.helper.make_node(
        'ReduceSum',
        inputs=['data', 'axes'],
        outputs=['reduced'],
        keepdims=keepdims)

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
    reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)
    #print(reduced)
    #[[[4., 6.]]
    # [[12., 14.]]
    # [[20., 22.]]]

    expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_keepdims_example')

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)

    expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_keepdims_random')

**default_axes_keepdims**

::

    shape = [3, 2, 2]
    axes = np.array([], dtype=np.int64)
    keepdims = 1

    node = onnx.helper.make_node(
        'ReduceSum',
        inputs=['data', 'axes'],
        outputs=['reduced'],
        keepdims=keepdims)

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
    reduced = np.sum(data, axis=None, keepdims=keepdims == 1)
    #print(reduced)
    #[[[78.]]]

    expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_default_axes_keepdims_example')

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.sum(data, axis=None, keepdims=keepdims == 1)

    expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_default_axes_keepdims_random')

**negative_axes_keepdims**

::

    shape = [3, 2, 2]
    axes = np.array([-2], dtype=np.int64)
    keepdims = 1

    node = onnx.helper.make_node(
        'ReduceSum',
        inputs=['data', 'axes'],
        outputs=['reduced'],
        keepdims=keepdims)

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
    reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)
    # print(reduced)
    #[[[4., 6.]]
    # [[12., 14.]]
    # [[20., 22.]]]

    expect(node, inputs=[data, axes], outputs=[reduced],
           name='test_reduce_sum_negative_axes_keepdims_example')

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.sum(data, axis=tuple(
        axes.tolist()), keepdims=keepdims == 1)

    expect(node, inputs=[data, axes], outputs=[reduced],
           name='test_reduce_sum_negative_axes_keepdims_random')

**empty_axes_input_noop**

::

    shape = [3, 2, 2]
    keepdims = 1

    node = onnx.helper.make_node(
        'ReduceSum',
        inputs=['data', 'axes'],
        outputs=['reduced'],
        keepdims=keepdims,
        noop_with_empty_axes=True)

    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
    axes = np.array([], dtype=np.int64)
    reduced = np.array(data)
    #print(reduced)
    #[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]

    expect(node, inputs=[data, axes], outputs=[reduced],
           name='test_reduce_sum_empty_axes_input_noop_example')

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.array(data)

    expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_negative_axes_keepdims_random')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Computes the sum of the input tensor's element along the provided axes. The resulting</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Computes the sum of the input tensor's element along the provided axes. The resulting</code></td></tr>
    <tr style="1px solid black;"><td><code>1</code></td><td><code>1</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>tensor has the same rank as the input if keepdims equals 1. If keepdims equal<span style="color:#196F3D;">s</span> 0, then</code></td></tr>
    <tr style="1px solid black;"><td><code>2</code></td><td><code>2</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">the result<span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">d</span> tensor ha<span style="color:#BA4A00;">v</span><span style="color:#BA4A00;">e</span> the reduced dimension pruned.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>the result<span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span> tensor ha<span style="color:#196F3D;">s</span> the reduced dimension pruned.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The above behavior is similar to numpy, with the exception that numpy defaults keepdims to</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The above behavior is similar to numpy, with the exception that numpy defaults keepdims to</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">False instead of True.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">False instead of True.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">9</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **keepdims**:</code></td></tr>
    <tr style="1px solid black;"><td><code>9</code></td><td><code>10</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">*</span> <span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">*</span>a<span style="color:#BA4A00;">x</span>es<span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">:</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code> <span style="color:#196F3D;"> </span><span style="color:#196F3D;">K</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">f</span>a<span style="color:#196F3D;">u</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">1</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">m</span>e<span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span>s<span style="color:#196F3D;"> </span><span style="color:#196F3D;">k</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span></code></td></tr>
    <tr style="1px solid black;"><td><code>10</code></td><td><code>11</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  <span style="color:#BA4A00;">A</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">l</span>is<span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span>i<span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span>on<span style="color:#BA4A00;">g</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">w</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">e</span>. <span style="color:#BA4A00;">T</span><span style="color:#BA4A00;">h</span>e<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">e</span>fault is <span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">e</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">d</span>i<span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span>sion. <span style="color:#196F3D;">D</span>efault <span style="color:#196F3D;">v</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span>is <span style="color:#196F3D;">1</span><span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td><code>11</code></td><td><code>12</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"> <span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">v</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">e</span>n<span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">i</span>o<span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span>o<span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span>the<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span>p<span style="color:#BA4A00;">u</span>t<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">.</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">A</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">r</span>a<span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">g</span>e<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span>s<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">[</span><span style="color:#BA4A00;">-</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">,</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">*</span> <span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span>noo<span style="color:#196F3D;">p</span><span style="color:#196F3D;">_</span><span style="color:#196F3D;">w</span><span style="color:#196F3D;">i</span>th<span style="color:#196F3D;">_</span>e<span style="color:#196F3D;">m</span>pt<span style="color:#196F3D;">y</span><span style="color:#196F3D;">_</span>a<span style="color:#196F3D;">x</span>es<span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">:</span></code></td></tr>
    <tr style="1px solid black;"><td><code>12</code></td><td><code>13</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  <span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">-</span><span style="color:#BA4A00;">1</span><span style="color:#BA4A00;">]</span> <span style="color:#BA4A00;">w</span>h<span style="color:#BA4A00;">e</span>re <span style="color:#BA4A00;">r</span> <span style="color:#BA4A00;">=</span> <span style="color:#BA4A00;">r</span>a<span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">k</span><span style="color:#BA4A00;">(</span><span style="color:#BA4A00;">d</span>ata<span style="color:#BA4A00;">)</span><span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">D</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span> <span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span>h<span style="color:#196F3D;">a</span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span>r<span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">'</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">x</span>e<span style="color:#196F3D;">s</span><span style="color:#196F3D;">'</span> <span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span> <span style="color:#196F3D;">e</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;">.</span> <span style="color:#196F3D;">D</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">f</span>a<span style="color:#196F3D;">u</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">h</span>a<span style="color:#196F3D;">v</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">w</span><span style="color:#196F3D;">i</span>t<span style="color:#196F3D;">h</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">'</span><span style="color:#196F3D;">f</span>a<span style="color:#196F3D;">l</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">'</span></code></td></tr>
    <tr style="1px solid black;"><td><code>13</code></td><td><code>14</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">*</span> <span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">k</span>eepdi<span style="color:#BA4A00;">m</span>s<span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">:</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code> <span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">r</span>e<span style="color:#196F3D;">d</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">c</span>e<span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">W</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">m</span>p<span style="color:#196F3D;">t</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span>d<span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span>is<span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">t</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">14</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  Keep the reduced dimension or not, default 1 means keep reduced</code></td><td></td></tr>
    <tr style="1px solid black;"><td><code>15</code></td><td><code>15</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  <span style="color:#BA4A00;">d</span>i<span style="color:#BA4A00;">m</span>ension<span style="color:#BA4A00;">.</span> <span style="color:#BA4A00;">D</span>e<span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">a</span>u<span style="color:#BA4A00;">l</span>t <span style="color:#BA4A00;">v</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">u</span>e<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span>s<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">1</span><span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span>i<span style="color:#196F3D;">n</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span>ens<span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">w</span>i<span style="color:#196F3D;">l</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">n</span>o<span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;">a</span>n<span style="color:#196F3D;">d</span> <span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span>e<span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span>ut<span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span> <span style="color:#196F3D;">t</span>e<span style="color:#196F3D;">n</span>s<span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">16</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  would be equivalent to input tensor. Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">20</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">Between 1 and 2 inputs.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">21</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">23</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  An input tensor.</code></td></tr>
    <tr style="1px solid black;"><td><code>20</code></td><td><code>24</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  <span style="color:#BA4A00;">A</span>n <span style="color:#BA4A00;">i</span>n<span style="color:#BA4A00;">p</span>u<span style="color:#BA4A00;">t</span> tensor<span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">*</span> <span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span> <span style="color:#196F3D;">(</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span>n<span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">,</span> <span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">e</span>n<span style="color:#196F3D;">e</span><span style="color:#196F3D;">o</span>u<span style="color:#196F3D;">s</span><span style="color:#196F3D;">)</span> <span style="color:#196F3D;">-</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span>tensor<span style="color:#196F3D;">(</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">6</span><span style="color:#196F3D;">4</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">:</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">25</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Optional input list of integers, along which to reduce. The default</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">26</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  is to reduce over all the dimensions of the input tensor if</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">27</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  'noop_with_empty_axes' is false, else act as an Identity op when</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">28</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  'noop_with_empty_axes' is true. Accepted range is [-r, r-1] where r</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">29</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  = rank(data).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **reduced** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **reduced** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Reduced output tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Reduced output tensor.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">39</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(bfloat16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to high-precision numeric tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to high-precision numeric tensors.</code></td></tr>
    </table>

.. _l-onnx-op-reducesum-11:

ReduceSum - 11
==============

**Version**

* **name**: `ReduceSum (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSum>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

Computes the sum of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.

**Attributes**

* **axes**:
  A list of integers, along which to reduce. The default is to reduce
  over all the dimensions of the input tensor. Accepted range is [-r,
  r-1] where r = rank(data).
* **keepdims**:
  Keep the reduced dimension or not, default 1 means keep reduced
  dimension. Default value is ``1``.

**Inputs**

* **data** (heterogeneous) - **T**:
  An input tensor.

**Outputs**

* **reduced** (heterogeneous) - **T**:
  Reduced output tensor.

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int32),
  tensor(int64),
  tensor(uint32),
  tensor(uint64)
  ):
  Constrain input and output types to high-precision numeric tensors.

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Computes the sum of the input tensor's element along the provided axes. The resulting</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Computes the sum of the input tensor's element along the provided axes. The resulting</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the resulted tensor have the reduced dimension pruned.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the resulted tensor have the reduced dimension pruned.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The above behavior is similar to numpy, with the exception that numpy defaults keepdims to</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The above behavior is similar to numpy, with the exception that numpy defaults keepdims to</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">False instead of True.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">False instead of True.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **axes**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **axes**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A list of integers, along which to reduce. The default is to reduce</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A list of integers, along which to reduce. The default is to reduce</code></td></tr>
    <tr style="1px solid black;"><td><code>11</code></td><td><code>11</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  over all the dimensions of the input tensor.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  over all the dimensions of the input tensor.<span style="color:#196F3D;"> </span><span style="color:#196F3D;">A</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">[</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">,</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">12</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  r-1] where r = rank(data).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **keepdims**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **keepdims**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Keep the reduced dimension or not, default 1 means keep reduced</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Keep the reduced dimension or not, default 1 means keep reduced</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimension. Default value is 1.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimension. Default value is 1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  An input tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  An input tensor.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **reduced** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **reduced** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Reduced output tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Reduced output tensor.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to high-precision numeric tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to high-precision numeric tensors.</code></td></tr>
    </table>

.. _l-onnx-op-reducesum-1:

ReduceSum - 1
=============

**Version**

* **name**: `ReduceSum (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSum>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1**.

**Summary**

Computes the sum of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.

**Attributes**

* **axes**:
  A list of integers, along which to reduce. The default is to reduce
  over all the dimensions of the input tensor.
* **keepdims**:
  Keep the reduced dimension or not, default 1 means keep reduced
  dimension. Default value is ``1``.

**Inputs**

* **data** (heterogeneous) - **T**:
  An input tensor.

**Outputs**

* **reduced** (heterogeneous) - **T**:
  Reduced output tensor.

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int32),
  tensor(int64),
  tensor(uint32),
  tensor(uint64)
  ):
  Constrain input and output types to high-precision numeric tensors.
