
.. _l-onnx-doc-CumSum:

======
CumSum
======

.. contents::
    :local:


.. _l-onnx-op-cumsum-14:

CumSum - 14
===========

**Version**

* **name**: `CumSum (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#CumSum>`_
* **domain**: **main**
* **since_version**: **14**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 14**.

**Summary**

Performs cumulative sum of the input elements along the given axis.
By default, it will do the sum inclusively meaning the first element is copied as is.
Through an `exclusive` attribute, this behavior can change to exclude the first element.
It can also perform summation in the opposite direction of the axis. For that, set `reverse` attribute to 1.

Example:
::

    input_x = [1, 2, 3]
    axis=0
    output = [1, 3, 6]
    exclusive=1
    output = [0, 1, 3]
    exclusive=0
    reverse=1
    output = [6, 5, 3]
    exclusive=1
    reverse=1
    output = [5, 3, 0]

**Attributes**

* **exclusive**:
  If set to 1 will return exclusive sum in which the top element is
  not included. In other terms, if set to 1, the j-th output element
  would be the sum of the first (j-1) elements. Otherwise, it would be
  the sum of the first j elements. Default value is ``0``.
* **reverse**:
  If set to 1 will perform the sums in reverse direction. Default value is ``0``.

**Inputs**

* **x** (heterogeneous) - **T**:
  An input tensor that is to be processed.
* **axis** (heterogeneous) - **T2**:
  A 0-D tensor. Must be in the range [-rank(x), rank(x)-1]. Negative
  value means counting dimensions from the back.

**Outputs**

* **y** (heterogeneous) - **T**:
  Output tensor of the same type as 'x' with cumulative sums of the
  x's elements

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
* **T2** in (
  tensor(int32),
  tensor(int64)
  ):
  axis tensor can be int32 or int64 only

**Examples**

**cumsum_1d**

::

    node = onnx.helper.make_node(
        'CumSum',
        inputs=['x', 'axis'],
        outputs=['y']
    )
    x = np.array([1., 2., 3., 4., 5.]).astype(np.float64)
    axis = np.int32(0)
    y = np.array([1., 3., 6., 10., 15.]).astype(np.float64)
    expect(node, inputs=[x, axis], outputs=[y],
           name='test_cumsum_1d')

**cumsum_1d_exclusive**

::

    node = onnx.helper.make_node(
        'CumSum',
        inputs=['x', 'axis'],
        outputs=['y'],
        exclusive=1
    )
    x = np.array([1., 2., 3., 4., 5.]).astype(np.float64)
    axis = np.int32(0)
    y = np.array([0., 1., 3., 6., 10.]).astype(np.float64)
    expect(node, inputs=[x, axis], outputs=[y],
           name='test_cumsum_1d_exclusive')

**cumsum_1d_reverse**

::

    node = onnx.helper.make_node(
        'CumSum',
        inputs=['x', 'axis'],
        outputs=['y'],
        reverse=1
    )
    x = np.array([1., 2., 3., 4., 5.]).astype(np.float64)
    axis = np.int32(0)
    y = np.array([15., 14., 12., 9., 5.]).astype(np.float64)
    expect(node, inputs=[x, axis], outputs=[y],
           name='test_cumsum_1d_reverse')

**cumsum_1d_reverse_exclusive**

::

    node = onnx.helper.make_node(
        'CumSum',
        inputs=['x', 'axis'],
        outputs=['y'],
        reverse=1,
        exclusive=1
    )
    x = np.array([1., 2., 3., 4., 5.]).astype(np.float64)
    axis = np.int32(0)
    y = np.array([14., 12., 9., 5., 0.]).astype(np.float64)
    expect(node, inputs=[x, axis], outputs=[y],
           name='test_cumsum_1d_reverse_exclusive')

**cumsum_2d_axis_0**

::

    node = onnx.helper.make_node(
        'CumSum',
        inputs=['x', 'axis'],
        outputs=['y'],
    )
    x = np.array([1., 2., 3., 4., 5., 6.]).astype(np.float64).reshape((2, 3))
    axis = np.int32(0)
    y = np.array([1., 2., 3., 5., 7., 9.]).astype(np.float64).reshape((2, 3))
    expect(node, inputs=[x, axis], outputs=[y],
           name='test_cumsum_2d_axis_0')

**cumsum_2d_axis_1**

::

    node = onnx.helper.make_node(
        'CumSum',
        inputs=['x', 'axis'],
        outputs=['y'],
    )
    x = np.array([1., 2., 3., 4., 5., 6.]).astype(np.float64).reshape((2, 3))
    axis = np.int32(1)
    y = np.array([1., 3., 6., 4., 9., 15.]).astype(np.float64).reshape((2, 3))
    expect(node, inputs=[x, axis], outputs=[y],
           name='test_cumsum_2d_axis_1')

**cumsum_2d_negative_axis**

::

    node = onnx.helper.make_node(
        'CumSum',
        inputs=['x', 'axis'],
        outputs=['y'],
    )
    x = np.array([1., 2., 3., 4., 5., 6.]).astype(np.float64).reshape((2, 3))
    axis = np.int32(-1)
    y = np.array([1., 3., 6., 4., 9., 15.]).astype(np.float64).reshape((2, 3))
    expect(node, inputs=[x, axis], outputs=[y],
           name='test_cumsum_2d_negative_axis')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Performs cumulative sum of the input elements along the given axis.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Performs cumulative sum of the input elements along the given axis.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">By default, it will do the sum inclusively meaning the first element is copied as is.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">By default, it will do the sum inclusively meaning the first element is copied as is.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Through an exclusive attribute, this behavior can change to exclude the first element.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Through an exclusive attribute, this behavior can change to exclude the first element.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">It can also perform summation in the opposite direction of the axis. For that, set reverse attribute to 1.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">It can also perform summation in the opposite direction of the axis. For that, set reverse attribute to 1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input_x = [1, 2, 3]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input_x = [1, 2, 3]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    axis=0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    axis=0</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    output = [1, 3, 6]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    output = [1, 3, 6]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    exclusive=1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    exclusive=1</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    output = [0, 1, 3]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    output = [0, 1, 3]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    exclusive=0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    exclusive=0</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    reverse=1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    reverse=1</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    output = [6, 5, 3]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    output = [6, 5, 3]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    exclusive=1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    exclusive=1</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    reverse=1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    reverse=1</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    output = [5, 3, 0]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    output = [5, 3, 0]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **exclusive**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **exclusive**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  If set to 1 will return exclusive sum in which the top element is</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  If set to 1 will return exclusive sum in which the top element is</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  not included. In other terms, if set to 1, the j-th output element</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  not included. In other terms, if set to 1, the j-th output element</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  would be the sum of the first (j-1) elements. Otherwise, it would be</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  would be the sum of the first (j-1) elements. Otherwise, it would be</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the sum of the first j elements. Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the sum of the first j elements. Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **reverse**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **reverse**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  If set to 1 will perform the sums in reverse direction. Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  If set to 1 will perform the sums in reverse direction. Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **x** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **x** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  An input tensor that is to be processed.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  An input tensor that is to be processed.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **axis** (heterogeneous) - **T2**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **axis** (heterogeneous) - **T2**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A 0-D tensor. Must be in the range [-rank(x), rank(x)-1]. Negative</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A 0-D tensor. Must be in the range [-rank(x), rank(x)-1]. Negative</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  value means counting dimensions from the back.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  value means counting dimensions from the back.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **y** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **y** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor of the same type as 'x' with cumulative sums of the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor of the same type as 'x' with cumulative sums of the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x's elements</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x's elements</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">47</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(bfloat16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">50</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td><code>54</code></td><td><code>56</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  <span style="color:#BA4A00;">I</span>nput <span style="color:#BA4A00;">c</span>an <span style="color:#BA4A00;">b</span>e o<span style="color:#BA4A00;">f</span> <span style="color:#BA4A00;">a</span>n<span style="color:#BA4A00;">y</span> tensor<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">y</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">e</span>.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">C</span><span style="color:#196F3D;">o</span>n<span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span>put an<span style="color:#196F3D;">d</span> <span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;">p</span>e<span style="color:#196F3D;">s</span> <span style="color:#196F3D;">t</span>o <span style="color:#196F3D;">h</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span>n <span style="color:#196F3D;">n</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;"> </span>tensor<span style="color:#196F3D;">s</span>.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T2** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T2** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  axis tensor can be int32 or int64 only</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  axis tensor can be int32 or int64 only</code></td></tr>
    </table>

.. _l-onnx-op-cumsum-11:

CumSum - 11
===========

**Version**

* **name**: `CumSum (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#CumSum>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

Performs cumulative sum of the input elements along the given axis.
By default, it will do the sum inclusively meaning the first element is copied as is.
Through an `exclusive` attribute, this behavior can change to exclude the first element.
It can also perform summation in the opposite direction of the axis. For that, set `reverse` attribute to 1.

Example:
::

    input_x = [1, 2, 3]
    axis=0
    output = [1, 3, 6]
    exclusive=1
    output = [0, 1, 3]
    exclusive=0
    reverse=1
    output = [6, 5, 3]
    exclusive=1
    reverse=1
    output = [5, 3, 0]

**Attributes**

* **exclusive**:
  If set to 1 will return exclusive sum in which the top element is
  not included. In other terms, if set to 1, the j-th output element
  would be the sum of the first (j-1) elements. Otherwise, it would be
  the sum of the first j elements. Default value is ``0``.
* **reverse**:
  If set to 1 will perform the sums in reverse direction. Default value is ``0``.

**Inputs**

* **x** (heterogeneous) - **T**:
  An input tensor that is to be processed.
* **axis** (heterogeneous) - **T2**:
  A 0-D tensor. Must be in the range [-rank(x), rank(x)-1]. Negative
  value means counting dimensions from the back.

**Outputs**

* **y** (heterogeneous) - **T**:
  Output tensor of the same type as 'x' with cumulative sums of the
  x's elements

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(int32),
  tensor(int64),
  tensor(uint32),
  tensor(uint64)
  ):
  Input can be of any tensor type.
* **T2** in (
  tensor(int32),
  tensor(int64)
  ):
  axis tensor can be int32 or int64 only
