
.. _l-onnx-doc-Scatter:

=======
Scatter
=======

.. contents::
    :local:


.. _l-onnx-op-scatter-11:

Scatter - 11
============

**Version**

* **name**: `Scatter (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Scatter>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been deprecated
**since version 11**.

**Summary**

This operator is deprecated. Please use ScatterElements, which provides the same functionality.

Scatter takes three inputs `data`, `updates`, and `indices` of the same
rank r >= 1 and an optional attribute axis that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). The output of the operation
is produced by creating a copy of the input `data`, and then updating its value
to values specified by `updates` at specific index positions specified by
`indices`. Its output shape is the same as the shape of `data`.

For each entry in `updates`, the target index in `data` is obtained by combining
the corresponding entry in `indices` with the index of the entry itself: the
index-value for dimension = axis is obtained from the value of the corresponding
entry in `indices` and the index-value for dimension != axis is obtained from the
index of the entry itself.

For instance, in a 2-D tensor case, the update corresponding to the [i][j] entry
is performed as below:
::

      output[indices[i][j]][j] = updates[i][j] if axis = 0,
      output[i][indices[i][j]] = updates[i][j] if axis = 1,

This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.

Example 1:
::

      data = [
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
      ]
      indices = [
          [1, 0, 2],
          [0, 2, 1],
      ]
      updates = [
          [1.0, 1.1, 1.2],
          [2.0, 2.1, 2.2],
      ]
      output = [
          [2.0, 1.1, 0.0]
          [1.0, 0.0, 2.2]
          [0.0, 2.1, 1.2]
      ]

Example 2:
::

      data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
      indices = [[1, 3]]
      updates = [[1.1, 2.1]]
      axis = 1
      output = [[1.0, 1.1, 3.0, 2.1, 5.0]]

**Attributes**

* **axis**:
  Which axis to scatter on. Negative value means counting dimensions
  from the back. Accepted range is [-r, r-1] where r = rank(data). Default value is ``0``.

**Inputs**

* **data** (heterogeneous) - **T**:
  Tensor of rank r >= 1.
* **indices** (heterogeneous) - **Tind**:
  Tensor of int32/int64 indices, of r >= 1 (same rank as input). All
  index values are expected to be within bounds [-s, s-1] along axis
  of size s. It is an error if any of the index values are out of
  bounds.
* **updates** (heterogeneous) - **T**:
  Tensor of rank r >=1 (same rank and shape as indices)

**Outputs**

* **output** (heterogeneous) - **T**:
  Tensor of rank r >= 1 (same rank as input).

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
  Input and output types can be of any tensor type.
* **Tind** in (
  tensor(int32),
  tensor(int64)
  ):
  Constrain indices to integer types

**Examples**

**scatter_without_axis**

::

    node = onnx.helper.make_node(
        'Scatter',
        inputs=['data', 'indices', 'updates'],
        outputs=['y'],
    )
    data = np.zeros((3, 3), dtype=np.float32)
    indices = np.array([[1, 0, 2], [0, 2, 1]], dtype=np.int64)
    updates = np.array([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], dtype=np.float32)

    y = scatter(data, indices, updates)
    # print(y) produces
    # [[2.0, 1.1, 0.0],
    #  [1.0, 0.0, 2.2],
    #  [0.0, 2.1, 1.2]]

    expect(node, inputs=[data, indices, updates], outputs=[y],
           name='test_scatter_without_axis', opset_imports=[helper.make_opsetid("", 10)])

**scatter_with_axis**

::

    axis = 1
    node = onnx.helper.make_node(
        'Scatter',
        inputs=['data', 'indices', 'updates'],
        outputs=['y'],
        axis=axis,
    )
    data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    indices = np.array([[1, 3]], dtype=np.int64)
    updates = np.array([[1.1, 2.1]], dtype=np.float32)

    y = scatter(data, indices, updates, axis=axis)
    # print(y) produces
    # [[1.0, 1.1, 3.0, 2.1, 5.0]]

    expect(node, inputs=[data, indices, updates], outputs=[y],
           name='test_scatter_with_axis', opset_imports=[helper.make_opsetid("", 10)])

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">0</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">Given data, updates and indices input tensors of rank r >= 1, write the values provided by updates</code></td><td></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">0</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">This operator is deprecated. Please use ScatterElements, which provides the same functionality.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">1</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">2</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">Scatter takes three inputs data, updates, and indices of the same</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">3</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">rank r >= 1 and an optional attribute axis that identifies an axis of data</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">4</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">(by default, the outer-most axis, that is axis 0). The output of the operation</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">5</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">is produced by creating a copy of the input data, and then updating its value</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">6</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">to values specified by updates at specific index positions specified by</code></td></tr>
    <tr style="1px solid black;"><td><code>1</code></td><td><code>7</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">in<span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">o</span> t<span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span> <span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">s</span>t<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span>put<span style="color:#BA4A00;">,</span> <span style="color:#BA4A00;">d</span>a<span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">,</span> <span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">x</span>is <span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">m</span>e<span style="color:#BA4A00;">n</span>s<span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">d</span>a<span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">(</span><span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">y</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">-</span>m<span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">n</span>e as <span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">x</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">=</span><span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">)</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span>t <span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">e</span>spo<span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">g</span> <span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span>d<span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">s</span>.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>in<span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">.</span> <span style="color:#196F3D;">I</span>t<span style="color:#196F3D;">s</span> <span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span>tput <span style="color:#196F3D;">s</span><span style="color:#196F3D;">h</span>a<span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span> is <span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span>e<span style="color:#196F3D;"> </span>same as t<span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span> s<span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span>p<span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span>o<span style="color:#196F3D;">f</span> d<span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span>.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">8</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td><code>2</code></td><td><code>9</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">For each entry in updates, the target index in data is <span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">c</span>i<span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">i</span>ed by co<span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">o</span>n<span style="color:#BA4A00;">d</span>ing<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">y</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">s</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>For each entry in updates, the target index in data is <span style="color:#196F3D;">o</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span>i<span style="color:#196F3D;">n</span>ed by co<span style="color:#196F3D;">m</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">i</span>ning</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">10</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">the corresponding entry in indices with the index of the entry itself: the</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">11</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">index-value for dimension = axis is obtained from the value of the corresponding</code></td></tr>
    <tr style="1px solid black;"><td><code>3</code></td><td><code>12</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">o</span>r <span style="color:#BA4A00;">d</span>i<span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">e</span>n<span style="color:#BA4A00;">s</span>i<span style="color:#BA4A00;">o</span>n<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">=</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">x</span>is<span style="color:#BA4A00;">,</span> and index<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">o</span>u<span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">c</span>e for dimension != axis<span style="color:#BA4A00;">.</span> <span style="color:#BA4A00;">F</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;"> </span>i<span style="color:#BA4A00;">n</span>stan<span style="color:#BA4A00;">c</span>e<span style="color:#BA4A00;">,</span> <span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span> <span style="color:#BA4A00;">a</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">2</span><span style="color:#BA4A00;">-</span><span style="color:#BA4A00;">D</span><span style="color:#BA4A00;"> </span>te<span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">,</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span>r<span style="color:#196F3D;">y</span> in<span style="color:#196F3D;"> </span>in<span style="color:#196F3D;">d</span>i<span style="color:#196F3D;">c</span><span style="color:#196F3D;">e</span>s and <span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span>index<span style="color:#196F3D;">-</span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span>ue for dimension != axis is<span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">b</span>ta<span style="color:#196F3D;">i</span>ne<span style="color:#196F3D;">d</span> <span style="color:#196F3D;">f</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">m</span> t<span style="color:#196F3D;">h</span>e</code></td></tr>
    <tr style="1px solid black;"><td><code>4</code></td><td><code>13</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">[</span>ind<span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">c</span>e<span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">[</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">]</span><span style="color:#BA4A00;">[</span><span style="color:#BA4A00;">j</span><span style="color:#BA4A00;">]</span><span style="color:#BA4A00;">]</span><span style="color:#BA4A00;">[</span><span style="color:#BA4A00;">j</span><span style="color:#BA4A00;">]</span> <span style="color:#BA4A00;">=</span> <span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">a</span>te<span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">[</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">]</span><span style="color:#BA4A00;">[</span><span style="color:#BA4A00;">j</span><span style="color:#BA4A00;">]</span> <span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">f</span> <span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">x</span>i<span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">=</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">a</span>t<span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">[</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">]</span><span style="color:#BA4A00;">[</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">e</span>s<span style="color:#BA4A00;">[</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">]</span><span style="color:#BA4A00;">[</span><span style="color:#BA4A00;">j</span><span style="color:#BA4A00;">]</span><span style="color:#BA4A00;">]</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">=</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">t</span>e<span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">[</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">]</span><span style="color:#BA4A00;">[</span><span style="color:#BA4A00;">j</span><span style="color:#BA4A00;">]</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span>f<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">x</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">=</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">1</span><span style="color:#BA4A00;">,</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>inde<span style="color:#196F3D;">x</span> <span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span> t<span style="color:#196F3D;">h</span>e <span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">y</span> itse<span style="color:#196F3D;">l</span>f<span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">14</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">15</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">For instance, in a 2-D tensor case, the update corresponding to the [i][j] entry</code></td></tr>
    <tr style="1px solid black;"><td><code>5</code></td><td><code>16</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">w</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span>i <span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">j</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">o</span>p<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">t</span>er<span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span>fr<span style="color:#BA4A00;">o</span>m<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">0</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span>e <span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">e</span>s<span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">v</span><span style="color:#BA4A00;">e</span> <span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">z</span>e<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">-</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">1</span><span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>i<span style="color:#196F3D;">s</span> perf<span style="color:#196F3D;">o</span>rme<span style="color:#196F3D;">d</span> <span style="color:#196F3D;">a</span>s <span style="color:#196F3D;">b</span>e<span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">w</span><span style="color:#196F3D;">:</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">17</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">::</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">18</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td><code>6</code></td><td><code>19</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">E</span><span style="color:#BA4A00;">x</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">m</span>p<span style="color:#BA4A00;">l</span>e <span style="color:#BA4A00;">1</span><span style="color:#BA4A00;">:</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span>p<span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">[</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">c</span>e<span style="color:#196F3D;">s</span><span style="color:#196F3D;">[</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">]</span><span style="color:#196F3D;">[</span><span style="color:#196F3D;">j</span><span style="color:#196F3D;">]</span><span style="color:#196F3D;">]</span><span style="color:#196F3D;">[</span><span style="color:#196F3D;">j</span><span style="color:#196F3D;">]</span> <span style="color:#196F3D;">=</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">[</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">]</span><span style="color:#196F3D;">[</span><span style="color:#196F3D;">j</span><span style="color:#196F3D;">]</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">=</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">0</span><span style="color:#196F3D;">,</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">7</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  data = [</code></td><td></td></tr>
    <tr style="1px solid black;"><td><code>8</code></td><td><code>20</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      [<span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">.</span><span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">,</span> <span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">.</span><span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">,</span> <span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">.</span><span style="color:#BA4A00;">0</span>],</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      <span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span>[<span style="color:#196F3D;">i</span><span style="color:#196F3D;">]</span><span style="color:#196F3D;">[</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">[</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">]</span><span style="color:#196F3D;">[</span><span style="color:#196F3D;">j</span><span style="color:#196F3D;">]</span><span style="color:#196F3D;">]</span> <span style="color:#196F3D;">=</span> <span style="color:#196F3D;">u</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">[</span><span style="color:#196F3D;">i</span>]<span style="color:#196F3D;">[</span><span style="color:#196F3D;">j</span><span style="color:#196F3D;">]</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">=</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">1</span>,</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">21</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td><code>9</code></td><td><code>22</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      <span style="color:#BA4A00;">[</span><span style="color:#BA4A00;">0</span>.<span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">,</span> <span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">.</span><span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">,</span> <span style="color:#BA4A00;">0</span>.<span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">]</span><span style="color:#BA4A00;">,</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">T</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span> <span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span> <span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span> <span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span> <span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span> <span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span> <span style="color:#196F3D;">G</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">E</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">s</span>. <span style="color:#196F3D;">I</span><span style="color:#196F3D;">t</span> <span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">T</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">'</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">S</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span>.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">23</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">24</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">Example 1:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">25</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">::</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">26</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">27</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">      data = [</code></td></tr>
    <tr style="1px solid black;"><td><code>10</code></td><td><code>28</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      [0.0, 0.0, 0.0],</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      <span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span>[0.0, 0.0, 0.0],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">11</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  ]</code></td><td></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">29</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">          [0.0, 0.0, 0.0],</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">30</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">          [0.0, 0.0, 0.0],</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">31</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">      ]</code></td></tr>
    <tr style="1px solid black;"><td><code>12</code></td><td><code>32</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  indices = [</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span>indices = [</code></td></tr>
    <tr style="1px solid black;"><td><code>13</code></td><td><code>33</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      [1, 0, 2],</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      <span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span>[1, 0, 2],</code></td></tr>
    <tr style="1px solid black;"><td><code>14</code></td><td><code>34</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      [0, 2, 1],</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      <span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span>[0, 2, 1],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">15</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  ]</code></td><td></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">35</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">      ]</code></td></tr>
    <tr style="1px solid black;"><td><code>16</code></td><td><code>36</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  updates = [</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span>updates = [</code></td></tr>
    <tr style="1px solid black;"><td><code>17</code></td><td><code>37</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      [1.0, 1.1, 1.2],</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      <span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span>[1.0, 1.1, 1.2],</code></td></tr>
    <tr style="1px solid black;"><td><code>18</code></td><td><code>38</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      [2.0, 2.1, 2.2],</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      <span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span>[2.0, 2.1, 2.2],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">19</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  ]</code></td><td></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">39</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">      ]</code></td></tr>
    <tr style="1px solid black;"><td><code>20</code></td><td><code>40</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  output = [</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span>output = [</code></td></tr>
    <tr style="1px solid black;"><td><code>21</code></td><td><code>41</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      [2.0, 1.1, 0.0]</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      <span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span>[2.0, 1.1, 0.0]</code></td></tr>
    <tr style="1px solid black;"><td><code>22</code></td><td><code>42</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      [1.0, 0.0, 2.2]</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      <span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span>[1.0, 0.0, 2.2]</code></td></tr>
    <tr style="1px solid black;"><td><code>23</code></td><td><code>43</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      [0.0, 2.1, 1.2]</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      <span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span>[0.0, 2.1, 1.2]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">24</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  ]</code></td><td></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">44</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">      ]</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">45</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 2:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 2:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">47</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">::</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">48</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td><code>26</code></td><td><code>49</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  data = [[1.0, 2.0, 3.0, 4.0, 5.0]]</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span>data = [[1.0, 2.0, 3.0, 4.0, 5.0]]</code></td></tr>
    <tr style="1px solid black;"><td><code>27</code></td><td><code>50</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  indices = [[1, 3]]</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span>indices = [[1, 3]]</code></td></tr>
    <tr style="1px solid black;"><td><code>28</code></td><td><code>51</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  updates = [[1.1, 2.1]]</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span>updates = [[1.1, 2.1]]</code></td></tr>
    <tr style="1px solid black;"><td><code>29</code></td><td><code>52</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  axis = 1</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span>axis = 1</code></td></tr>
    <tr style="1px solid black;"><td><code>30</code></td><td><code>53</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  output = [[1.0, 1.1, 3.0, 2.1, 5.0]]</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span>output = [[1.0, 1.1, 3.0, 2.1, 5.0]]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **axis**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **axis**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Which axis to scatter on. Negative value means counting dimensions</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Which axis to scatter on. Negative value means counting dimensions</code></td></tr>
    <tr style="1px solid black;"><td><code>36</code></td><td><code>59</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  from the back. Accepted range is [-r, r-1] Default value is 0.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  from the back. Accepted range is [-r, r-1] <span style="color:#196F3D;">w</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">=</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">k</span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span>Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of rank r >= 1.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of rank r >= 1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **indices** (heterogeneous) - **Tind**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **indices** (heterogeneous) - **Tind**:</code></td></tr>
    <tr style="1px solid black;"><td><code>43</code></td><td><code>66</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Tensor of int32/int64 indices, of r >= 1 (same rank as input).</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Tensor of int32/int64 indices, of r >= 1 (same rank as input).<span style="color:#196F3D;"> </span><span style="color:#196F3D;">A</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">l</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">67</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  index values are expected to be within bounds [-s, s-1] along axis</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">68</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  of size s. It is an error if any of the index values are out of</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">69</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  bounds.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **updates** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **updates** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of rank r >=1 (same rank and shape as indices)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of rank r >=1 (same rank and shape as indices)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of rank r >= 1 (same rank as input).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of rank r >= 1 (same rank as input).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input and output types can be of any tensor type.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input and output types can be of any tensor type.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Tind** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Tind** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain indices to integer types</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain indices to integer types</code></td></tr>
    </table>

.. _l-onnx-op-scatter-9:

Scatter - 9
===========

**Version**

* **name**: `Scatter (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Scatter>`_
* **domain**: **main**
* **since_version**: **9**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 9**.

**Summary**

Given `data`, `updates` and `indices` input tensors of rank r >= 1, write the values provided by `updates`
into the first input, `data`, along `axis` dimension of `data` (by default outer-most one as axis=0) at corresponding `indices`.
For each entry in `updates`, the target index in `data` is specified by corresponding entry in `indices`
for dimension = axis, and index in source for dimension != axis. For instance, in a 2-D tensor case,
data[indices[i][j]][j] = updates[i][j] if axis = 0, or data[i][indices[i][j]] = updates[i][j] if axis = 1,
where i and j are loop counters from 0 up to the respective size in `updates` - 1.
Example 1:
  data = [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
  ]
  indices = [
      [1, 0, 2],
      [0, 2, 1],
  ]
  updates = [
      [1.0, 1.1, 1.2],
      [2.0, 2.1, 2.2],
  ]
  output = [
      [2.0, 1.1, 0.0]
      [1.0, 0.0, 2.2]
      [0.0, 2.1, 1.2]
  ]
Example 2:
  data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
  indices = [[1, 3]]
  updates = [[1.1, 2.1]]
  axis = 1
  output = [[1.0, 1.1, 3.0, 2.1, 5.0]]

**Attributes**

* **axis**:
  Which axis to scatter on. Negative value means counting dimensions
  from the back. Accepted range is [-r, r-1] Default value is ``0``.

**Inputs**

* **data** (heterogeneous) - **T**:
  Tensor of rank r >= 1.
* **indices** (heterogeneous) - **Tind**:
  Tensor of int32/int64 indices, of r >= 1 (same rank as input).
* **updates** (heterogeneous) - **T**:
  Tensor of rank r >=1 (same rank and shape as indices)

**Outputs**

* **output** (heterogeneous) - **T**:
  Tensor of rank r >= 1 (same rank as input).

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
  Input and output types can be of any tensor type.
* **Tind** in (
  tensor(int32),
  tensor(int64)
  ):
  Constrain indices to integer types
