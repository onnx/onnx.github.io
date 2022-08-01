
.. _l-onnx-doc-Expand:

======
Expand
======

.. contents::
    :local:


.. _l-onnx-op-expand-13:

Expand - 13
===========

**Version**

* **name**: `Expand (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Expand>`_
* **domain**: **main**
* **since_version**: **13**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 13**.

**Summary**

Broadcast the input tensor following the given shape and the broadcast rule.
The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
Dimensions are right alignment;
Two corresponding dimensions must have the same value, or one of them is equal to 1.
Also, this operator is similar to numpy.broadcast_to(input, shape),
but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
or the shape.ndim < input.shape.ndim.

**Inputs**

* **input** (heterogeneous) - **T**:
  Input tensor
* **shape** (heterogeneous) - **tensor(int64)**:
  A 1-D tensor indicates the shape you want to expand to, following
  the broadcast rule

**Outputs**

* **output** (heterogeneous) - **T**:
  Output tensor

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
  Constrain input and output types to all tensors.

**Examples**

**dim_changed**

::

    node = onnx.helper.make_node(
        'Expand',
        inputs=['data', 'new_shape'],
        outputs=['expanded'],
    )
    shape = [3, 1]
    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    #print(data)
    #[[1.], [2.], [3.]]
    new_shape = [2, 1, 6]
    expanded = data * np.ones(new_shape, dtype=np.float32)
    #print(expanded)
    #[[[1., 1., 1., 1., 1., 1.],
    #  [2., 2., 2., 2., 2., 2.],
    #  [3., 3., 3., 3., 3., 3.]],
    #
    # [[1., 1., 1., 1., 1., 1.],
    #  [2., 2., 2., 2., 2., 2.],
    #  [3., 3., 3., 3., 3., 3.]]]
    new_shape = np.array(new_shape, dtype=np.int64)
    expect(node, inputs=[data, new_shape], outputs=[expanded],
           name='test_expand_dim_changed')

**dim_unchanged**

::

    node = onnx.helper.make_node(
        'Expand',
        inputs=['data', 'new_shape'],
        outputs=['expanded'],
    )
    shape = [3, 1]
    new_shape = [3, 4]
    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    #print(data)
    #[[1.], [2.], [3.]]
    expanded = np.tile(data, 4)
    #print(expanded)
    #[[1., 1., 1., 1.],
    # [2., 2., 2., 2.],
    # [3., 3., 3., 3.]]
    new_shape = np.array(new_shape, dtype=np.int64)
    expect(node, inputs=[data, new_shape], outputs=[expanded],
           name='test_expand_dim_unchanged')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Broadcast the input tensor following the given shape and the broadcast rule.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Broadcast the input tensor following the given shape and the broadcast rule.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Dimensions are right alignment;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Dimensions are right alignment;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Two corresponding dimensions must have the same value, or one of them is equal to 1.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Two corresponding dimensions must have the same value, or one of them is equal to 1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Also, this operator is similar to numpy.broadcast_to(input, shape),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Also, this operator is similar to numpy.broadcast_to(input, shape),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">or the shape.ndim < input.shape.ndim.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">or the shape.ndim < input.shape.ndim.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **input** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **input** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **shape** (heterogeneous) - **tensor(int64)**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **shape** (heterogeneous) - **tensor(int64)**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A 1-D tensor indicates the shape you want to expand to, following</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A 1-D tensor indicates the shape you want to expand to, following</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the broadcast rule</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the broadcast rule</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">25</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(bfloat16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to all tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to all tensors.</code></td></tr>
    </table>

.. _l-onnx-op-expand-8:

Expand - 8
==========

**Version**

* **name**: `Expand (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Expand>`_
* **domain**: **main**
* **since_version**: **8**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 8**.

**Summary**

Broadcast the input tensor following the given shape and the broadcast rule.
The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
Dimensions are right alignment;
Two corresponding dimensions must have the same value, or one of them is equal to 1.
Also, this operator is similar to numpy.broadcast_to(input, shape),
but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
or the shape.ndim < input.shape.ndim.

**Inputs**

* **input** (heterogeneous) - **T**:
  Input tensor
* **shape** (heterogeneous) - **tensor(int64)**:
  A 1-D tensor indicates the shape you want to expand to, following
  the broadcast rule

**Outputs**

* **output** (heterogeneous) - **T**:
  Output tensor

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
  Constrain input and output types to all tensors.
