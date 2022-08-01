
.. _l-onnx-doc-OneHot:

======
OneHot
======

.. contents::
    :local:


.. _l-onnx-op-onehot-11:

OneHot - 11
===========

**Version**

* **name**: `OneHot (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#OneHot>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

Produces a one-hot tensor based on inputs.
The locations represented by the index values in the 'indices' input tensor will have 'on_value'
and the other locations will have 'off_value' in the output tensor, where 'on_value' and 'off_value'
are specified as part of required input argument 'values', which is a two-element tensor of format
[off_value, on_value]. The rank of the output tensor will be one greater than the rank of the
input tensor. The additional dimension is for one-hot representation. The additional dimension will
be inserted at the position specified by 'axis'. If 'axis' is not specified then then additional
dimension will be inserted as the innermost dimension, i.e. axis=-1. The size of the additional
dimension is specified by required scalar input 'depth'. The type of the output tensor is the same
as the type of the 'values' input. Any entries in the 'indices' input tensor with values outside
the range [-depth, depth-1] will result in one-hot representation with all 'off_value' values in the
output tensor.

when axis = 0:
output[input[i, j, k], i, j, k] = 1 for all i, j, k and 0 otherwise.

when axis = -1:
output[i, j, k, input[i, j, k]] = 1 for all i, j, k and 0 otherwise.

**Attributes**

* **axis**:
  (Optional) Axis along which one-hot representation in added.
  Default: axis=-1. axis=-1 means that the additional dimension will
  be inserted as the innermost/last dimension in the output tensor.
  Negative value means counting dimensions from the back. Accepted
  range is [-r-1, r] where r = rank(indices). Default value is ``-1``.

**Inputs**

* **indices** (heterogeneous) - **T1**:
  Input tensor containing indices. Any entries in the 'indices' input
  tensor with values outside the range [-depth, depth-1] will result
  in one-hot representation with all 'off_value' values in the output
  tensor.In case 'indices' is of non-integer type, the values will be
  casted to int64 before use.
* **depth** (heterogeneous) - **T2**:
  Scalar specifying the number of classes in one-hot tensor. This is
  also the size of the one-hot dimension (specified by 'axis'
  attribute) added on in the output tensor. The values in the
  'indices' input tensor are expected to be in the range [-depth,
  depth-1]. In case 'depth' is of non-integer type, it will be casted
  to int64 before use.
* **values** (heterogeneous) - **T3**:
  Rank 1 tensor containing exactly two elements, in the format
  [off_value, on_value], where 'on_value' is the value used for
  filling locations specified in 'indices' input tensor, and
  'off_value' is the value used for filling locations other than those
  specified in 'indices' input tensor.

**Outputs**

* **output** (heterogeneous) - **T3**:
  Tensor of rank one greater than input tensor 'indices', i.e.
  rank(output) = rank(indices) + 1. The data type for the elements of
  the output tensor is the same as the type of input 'values' is used.

**Type Constraints**

* **T1** in (
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int16),
  tensor(int32),
  tensor(int64),
  tensor(int8),
  tensor(uint16),
  tensor(uint32),
  tensor(uint64),
  tensor(uint8)
  ):
  Constrain input to only numeric types.
* **T2** in (
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int16),
  tensor(int32),
  tensor(int64),
  tensor(int8),
  tensor(uint16),
  tensor(uint32),
  tensor(uint64),
  tensor(uint8)
  ):
  Constrain input to only numeric types.
* **T3** in (
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
  Constrain to any tensor type.

**Examples**

**without_axis**

::

    on_value = 5
    off_value = 2
    output_type = np.int32
    node = onnx.helper.make_node(
        'OneHot',
        inputs=['indices', 'depth', 'values'],
        outputs=['y']
    )
    indices = np.array([0, 7, 8], dtype=np.int64)
    depth = np.float32(12)
    values = np.array([off_value, on_value], dtype=output_type)
    y = one_hot(indices, depth, dtype=output_type)
    y = y * (on_value - off_value) + off_value
    expect(node, inputs=[indices, depth, values], outputs=[y], name='test_onehot_without_axis')

**with_axis**

::

    axisValue = 1
    on_value = 3
    off_value = 1
    output_type = np.float32
    node = onnx.helper.make_node(
        'OneHot',
        inputs=['indices', 'depth', 'values'],
        outputs=['y'],
        axis=axisValue
    )
    indices = np.array([[1, 9],
                        [2, 4]], dtype=np.float32)
    depth = np.float32(10)
    values = np.array([off_value, on_value], dtype=output_type)
    y = one_hot(indices, depth, axis=axisValue, dtype=output_type)
    y = y * (on_value - off_value) + off_value
    expect(node, inputs=[indices, depth, values], outputs=[y], name='test_onehot_with_axis')

**with_negative_indices**

::

    axisValue = 1
    on_value = 3
    off_value = 1
    output_type = np.float32
    node = onnx.helper.make_node(
        'OneHot',
        inputs=['indices', 'depth', 'values'],
        outputs=['y'],
        axis=axisValue
    )
    indices = np.array([0, -7, -8], dtype=np.int64)

    # print(y)
    # [[3. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    #  [1. 1. 1. 3. 1. 1. 1. 1. 1. 1.]
    #  [1. 1. 3. 1. 1. 1. 1. 1. 1. 1.]]

    depth = np.float32(10)
    values = np.array([off_value, on_value], dtype=output_type)
    y = one_hot(indices, depth, axis=axisValue, dtype=output_type)
    y = y * (on_value - off_value) + off_value
    expect(node, inputs=[indices, depth, values], outputs=[y], name='test_onehot_negative_indices')

**with_negative_axis**

::

    axisValue = -2
    on_value = 3
    off_value = 1
    output_type = np.float32
    node = onnx.helper.make_node(
        'OneHot',
        inputs=['indices', 'depth', 'values'],
        outputs=['y'],
        axis=axisValue
    )
    indices = np.array([[1, 9],
                        [2, 4]], dtype=np.float32)
    depth = np.float32(10)
    values = np.array([off_value, on_value], dtype=output_type)
    y = one_hot(indices, depth, axis=axisValue, dtype=output_type)
    y = y * (on_value - off_value) + off_value
    expect(node, inputs=[indices, depth, values], outputs=[y], name='test_onehot_with_negative_axis')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Produces a one-hot tensor based on inputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Produces a one-hot tensor based on inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The locations represented by the index values in the 'indices' input tensor will have 'on_value'</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The locations represented by the index values in the 'indices' input tensor will have 'on_value'</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">and the other locations will have 'off_value' in the output tensor, where 'on_value' and 'off_value'</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">and the other locations will have 'off_value' in the output tensor, where 'on_value' and 'off_value'</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">are specified as part of required input argument 'values', which is a two-element tensor of format</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">are specified as part of required input argument 'values', which is a two-element tensor of format</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">[off_value, on_value]. The rank of the output tensor will be one greater than the rank of the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">[off_value, on_value]. The rank of the output tensor will be one greater than the rank of the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">input tensor. The additional dimension is for one-hot representation. The additional dimension will</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">input tensor. The additional dimension is for one-hot representation. The additional dimension will</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">be inserted at the position specified by 'axis'. If 'axis' is not specified then then additional</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">be inserted at the position specified by 'axis'. If 'axis' is not specified then then additional</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">dimension will be inserted as the innermost dimension, i.e. axis=-1. The size of the additional</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">dimension will be inserted as the innermost dimension, i.e. axis=-1. The size of the additional</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">dimension is specified by required scalar input 'depth'. The type of the output tensor is the same</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">dimension is specified by required scalar input 'depth'. The type of the output tensor is the same</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">as the type of the 'values' input. Any entries in the 'indices' input tensor with values outside</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">as the type of the 'values' input. Any entries in the 'indices' input tensor with values outside</code></td></tr>
    <tr style="1px solid black;"><td><code>10</code></td><td><code>10</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">the range [<span style="color:#BA4A00;">0</span>, depth<span style="color:#BA4A00;">)</span> will result in one-hot representation with all 'off_value' values in the</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>the range [<span style="color:#196F3D;">-</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span>, depth<span style="color:#196F3D;">-</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">]</span> will result in one-hot representation with all 'off_value' values in the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">output tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">output tensor.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">13</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">when axis = 0:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">14</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">output[input[i, j, k], i, j, k] = 1 for all i, j, k and 0 otherwise.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">15</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">16</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">when axis = -1:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">17</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">output[i, j, k, input[i, j, k]] = 1 for all i, j, k and 0 otherwise.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">18</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **axis**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **axis**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (Optional) Axis along which one-hot representation in added.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (Optional) Axis along which one-hot representation in added.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Default: axis=-1. axis=-1 means that the additional dimension will</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Default: axis=-1. axis=-1 means that the additional dimension will</code></td></tr>
    <tr style="1px solid black;"><td><code>18</code></td><td><code>24</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  be inserted as the innermost/last dimension in the output tensor.<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">D</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">v</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">-</span><span style="color:#BA4A00;">1</span><span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  be inserted as the innermost/last dimension in the output tensor.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">25</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Negative value means counting dimensions from the back. Accepted</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">26</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  range is [-r-1, r] where r = rank(indices). Default value is -1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **indices** (heterogeneous) - **T1**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **indices** (heterogeneous) - **T1**:</code></td></tr>
    <tr style="1px solid black;"><td><code>23</code></td><td><code>31</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Input tensor containing indices. <span style="color:#BA4A00;">T</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span> <span style="color:#BA4A00;">v</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">u</span>es <span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">s</span>t <span style="color:#BA4A00;">b</span>e n<span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">-</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;">a</span>t<span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">v</span><span style="color:#BA4A00;">e</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Input tensor containing indices. <span style="color:#196F3D;">A</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">y</span> e<span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">e</span>s <span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span>t<span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span> <span style="color:#196F3D;">'</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">c</span>e<span style="color:#196F3D;">s</span><span style="color:#196F3D;">'</span> <span style="color:#196F3D;">i</span>n<span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span>t</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">24</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  integers. Any entries in the 'indices' input tensor with values</code></td><td></td></tr>
    <tr style="1px solid black;"><td><code>25</code></td><td><code>32</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  outside the range [<span style="color:#BA4A00;">0</span>, depth<span style="color:#BA4A00;">)</span> will result<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">-</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">n</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">s</span>o<span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">w</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span>u<span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span>tside the range [<span style="color:#196F3D;">-</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span>, depth<span style="color:#196F3D;">-</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">]</span> will result</code></td></tr>
    <tr style="1px solid black;"><td><code>26</code></td><td><code>33</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  with all 'off_value' values in the output<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">.</span><span style="color:#BA4A00;">I</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">'</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">'</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span>with all 'off_value' values in the output</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">34</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor.In case 'indices' is of non-integer type, the values will be</code></td></tr>
    <tr style="1px solid black;"><td><code>27</code></td><td><code>35</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  <span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">-</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">y</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">v</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">w</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span>casted to int64 before</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  casted to int64 before<span style="color:#196F3D;"> </span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">28</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  use.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **depth** (heterogeneous) - **T2**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **depth** (heterogeneous) - **T2**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Scalar specifying the number of classes in one-hot tensor. This is</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Scalar specifying the number of classes in one-hot tensor. This is</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  also the size of the one-hot dimension (specified by 'axis'</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  also the size of the one-hot dimension (specified by 'axis'</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  attribute) added on in the output tensor. The values in the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  attribute) added on in the output tensor. The values in the</code></td></tr>
    <tr style="1px solid black;"><td><code>33</code></td><td><code>40</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  'indices' input tensor are expected to be in the range [<span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span>depth<span style="color:#BA4A00;">)</span><span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  'indices' input tensor are expected to be in the range [<span style="color:#196F3D;">-</span>depth<span style="color:#196F3D;">,</span></code></td></tr>
    <tr style="1px solid black;"><td><code>34</code></td><td><code>41</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  In case 'depth' is of non-integer type, it will be casted<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">6</span><span style="color:#BA4A00;">4</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">]</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span>In case 'depth' is of non-integer type, it will be casted</code></td></tr>
    <tr style="1px solid black;"><td><code>35</code></td><td><code>42</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  before use.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">6</span><span style="color:#196F3D;">4</span><span style="color:#196F3D;"> </span>before use.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **values** (heterogeneous) - **T3**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **values** (heterogeneous) - **T3**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Rank 1 tensor containing exactly two elements, in the format</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Rank 1 tensor containing exactly two elements, in the format</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  [off_value, on_value], where 'on_value' is the value used for</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  [off_value, on_value], where 'on_value' is the value used for</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  filling locations specified in 'indices' input tensor, and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  filling locations specified in 'indices' input tensor, and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'off_value' is the value used for filling locations other than those</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'off_value' is the value used for filling locations other than those</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  specified in 'indices' input tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  specified in 'indices' input tensor.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T3**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T3**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of rank one greater than input tensor 'indices', i.e.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tensor of rank one greater than input tensor 'indices', i.e.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  rank(output) = rank(indices) + 1. The data type for the elements of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  rank(output) = rank(indices) + 1. The data type for the elements of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the output tensor is the same as the type of input 'values' is used.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the output tensor is the same as the type of input 'values' is used.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input to only numeric types.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input to only numeric types.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T2** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T2** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input to only numeric types.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input to only numeric types.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T3** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T3** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain to any tensor type.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain to any tensor type.</code></td></tr>
    </table>

.. _l-onnx-op-onehot-9:

OneHot - 9
==========

**Version**

* **name**: `OneHot (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#OneHot>`_
* **domain**: **main**
* **since_version**: **9**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 9**.

**Summary**

Produces a one-hot tensor based on inputs.
The locations represented by the index values in the 'indices' input tensor will have 'on_value'
and the other locations will have 'off_value' in the output tensor, where 'on_value' and 'off_value'
are specified as part of required input argument 'values', which is a two-element tensor of format
[off_value, on_value]. The rank of the output tensor will be one greater than the rank of the
input tensor. The additional dimension is for one-hot representation. The additional dimension will
be inserted at the position specified by 'axis'. If 'axis' is not specified then then additional
dimension will be inserted as the innermost dimension, i.e. axis=-1. The size of the additional
dimension is specified by required scalar input 'depth'. The type of the output tensor is the same
as the type of the 'values' input. Any entries in the 'indices' input tensor with values outside
the range [0, depth) will result in one-hot representation with all 'off_value' values in the
output tensor.

**Attributes**

* **axis**:
  (Optional) Axis along which one-hot representation in added.
  Default: axis=-1. axis=-1 means that the additional dimension will
  be inserted as the innermost/last dimension in the output tensor. Default value is ``-1``.

**Inputs**

* **indices** (heterogeneous) - **T1**:
  Input tensor containing indices. The values must be non-negative
  integers. Any entries in the 'indices' input tensor with values
  outside the range [0, depth) will result in one-hot representation
  with all 'off_value' values in the output tensor.In case 'indices'
  is of non-integer type, the values will be casted to int64 before
  use.
* **depth** (heterogeneous) - **T2**:
  Scalar specifying the number of classes in one-hot tensor. This is
  also the size of the one-hot dimension (specified by 'axis'
  attribute) added on in the output tensor. The values in the
  'indices' input tensor are expected to be in the range [0, depth).
  In case 'depth' is of non-integer type, it will be casted to int64
  before use.
* **values** (heterogeneous) - **T3**:
  Rank 1 tensor containing exactly two elements, in the format
  [off_value, on_value], where 'on_value' is the value used for
  filling locations specified in 'indices' input tensor, and
  'off_value' is the value used for filling locations other than those
  specified in 'indices' input tensor.

**Outputs**

* **output** (heterogeneous) - **T3**:
  Tensor of rank one greater than input tensor 'indices', i.e.
  rank(output) = rank(indices) + 1. The data type for the elements of
  the output tensor is the same as the type of input 'values' is used.

**Type Constraints**

* **T1** in (
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int16),
  tensor(int32),
  tensor(int64),
  tensor(int8),
  tensor(uint16),
  tensor(uint32),
  tensor(uint64),
  tensor(uint8)
  ):
  Constrain input to only numeric types.
* **T2** in (
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int16),
  tensor(int32),
  tensor(int64),
  tensor(int8),
  tensor(uint16),
  tensor(uint32),
  tensor(uint64),
  tensor(uint8)
  ):
  Constrain input to only numeric types.
* **T3** in (
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
  Constrain to any tensor type.
