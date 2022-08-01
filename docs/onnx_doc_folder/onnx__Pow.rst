
.. _l-onnx-doc-Pow:

===
Pow
===

.. contents::
    :local:


.. _l-onnx-op-pow-15:

Pow - 15
========

**Version**

* **name**: `Pow (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pow>`_
* **domain**: **main**
* **since_version**: **15**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 15**.

**Summary**

Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check `Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>`_.

**Inputs**

* **X** (heterogeneous) - **T**:
  First operand, base of the exponent.
* **Y** (heterogeneous) - **T1**:
  Second operand, power of the exponent.

**Outputs**

* **Z** (heterogeneous) - **T**:
  Output tensor

**Type Constraints**

* **T** in (
  tensor(bfloat16),
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int32),
  tensor(int64)
  ):
  Constrain input X and output types to float/int tensors.
* **T1** in (
  tensor(bfloat16),
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
  Constrain input Y types to float/int tensors.

**Examples**

**pow_broadcast**

::

    node = onnx.helper.make_node(
        'Pow',
        inputs=['x', 'y'],
        outputs=['z'],
    )

    x = np.array([1, 2, 3]).astype(np.float32)
    y = np.array(2).astype(np.float32)
    z = pow(x, y)  # expected output [1., 4., 9.]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_pow_bcast_scalar')

    node = onnx.helper.make_node(
        'Pow',
        inputs=['x', 'y'],
        outputs=['z'],
    )
    x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    y = np.array([1, 2, 3]).astype(np.float32)
    # expected output [[1, 4, 27], [4, 25, 216]]
    z = pow(x, y)
    expect(node, inputs=[x, y], outputs=[z],
           name='test_pow_bcast_array')

**types**

::

    node = onnx.helper.make_node(
        'Pow',
        inputs=['x', 'y'],
        outputs=['z'],
    )

    x = np.array([1, 2, 3]).astype(np.float32)
    y = np.array([4, 5, 6]).astype(np.int64)
    z = pow(x, y)  # expected output [1., 32., 729.]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_pow_types_float32_int64')

    x = np.array([1, 2, 3]).astype(np.int64)
    y = np.array([4, 5, 6]).astype(np.float32)
    z = pow(x, y)  # expected output [1, 32, 729]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_pow_types_int64_float32')

    x = np.array([1, 2, 3]).astype(np.float32)
    y = np.array([4, 5, 6]).astype(np.int32)
    z = pow(x, y)  # expected output [1., 32., 729.]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_pow_types_float32_int32')

    x = np.array([1, 2, 3]).astype(np.int32)
    y = np.array([4, 5, 6]).astype(np.float32)
    z = pow(x, y)  # expected output [1, 32, 729]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_pow_types_int32_float32')

    x = np.array([1, 2, 3]).astype(np.float32)
    y = np.array([4, 5, 6]).astype(np.uint64)
    z = pow(x, y)  # expected output [1., 32., 729.]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_pow_types_float32_uint64')

    x = np.array([1, 2, 3]).astype(np.float32)
    y = np.array([4, 5, 6]).astype(np.uint32)
    z = pow(x, y)  # expected output [1., 32., 729.]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_pow_types_float32_uint32')

    x = np.array([1, 2, 3]).astype(np.int64)
    y = np.array([4, 5, 6]).astype(np.int64)
    z = pow(x, y)  # expected output [1, 32, 729]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_pow_types_int64_int64')

    x = np.array([1, 2, 3]).astype(np.int32)
    y = np.array([4, 5, 6]).astype(np.int32)
    z = pow(x, y)  # expected output [1, 32, 729]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_pow_types_int32_int32')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Pow takes input data (Tensor<T>) and exponent Tensor, and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Pow takes input data (Tensor<T>) and exponent Tensor, and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">produces one output data (Tensor<T>) where the function f(x) = x^exponent,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">produces one output data (Tensor<T>) where the function f(x) = x^exponent,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">is applied to the data tensor elementwise.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">is applied to the data tensor elementwise.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>_.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>_.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  First operand, base of the exponent.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  First operand, base of the exponent.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T1**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T1**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Second operand, power of the exponent.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Second operand, power of the exponent.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Z** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Z** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bfloat16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bfloat16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input X and output types to float/int tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input X and output types to float/int tensors.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">29</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(bfloat16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input Y types to float/int tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input Y types to float/int tensors.</code></td></tr>
    </table>

.. _l-onnx-op-pow-13:

Pow - 13
========

**Version**

* **name**: `Pow (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pow>`_
* **domain**: **main**
* **since_version**: **13**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 13**.

**Summary**

Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check `Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>`_.

**Inputs**

* **X** (heterogeneous) - **T**:
  First operand, base of the exponent.
* **Y** (heterogeneous) - **T1**:
  Second operand, power of the exponent.

**Outputs**

* **Z** (heterogeneous) - **T**:
  Output tensor

**Type Constraints**

* **T** in (
  tensor(bfloat16),
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int32),
  tensor(int64)
  ):
  Constrain input X and output types to float/int tensors.
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
  Constrain input Y types to float/int tensors.

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Pow takes input data (Tensor<T>) and exponent Tensor, and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Pow takes input data (Tensor<T>) and exponent Tensor, and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">produces one output data (Tensor<T>) where the function f(x) = x^exponent,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">produces one output data (Tensor<T>) where the function f(x) = x^exponent,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">is applied to the data tensor elementwise.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">is applied to the data tensor elementwise.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>_.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>_.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  First operand, base of the exponent.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  First operand, base of the exponent.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T1**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T1**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Second operand, power of the exponent.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Second operand, power of the exponent.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Z** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Z** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td><code>15</code></td><td><code>15</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Output tensor<span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Output tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">20</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(bfloat16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input X and output types to float/int tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input X and output types to float/int tensors.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input Y types to float/int tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input Y types to float/int tensors.</code></td></tr>
    </table>

.. _l-onnx-op-pow-12:

Pow - 12
========

**Version**

* **name**: `Pow (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pow>`_
* **domain**: **main**
* **since_version**: **12**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 12**.

**Summary**

Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check `Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>`_.

**Inputs**

* **X** (heterogeneous) - **T**:
  First operand, base of the exponent.
* **Y** (heterogeneous) - **T1**:
  Second operand, power of the exponent.

**Outputs**

* **Z** (heterogeneous) - **T**:
  Output tensor.

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int32),
  tensor(int64)
  ):
  Constrain input X and output types to float/int tensors.
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
  Constrain input Y types to float/int tensors.

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Pow takes input data (Tensor<T>) and exponent Tensor, and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Pow takes input data (Tensor<T>) and exponent Tensor, and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">produces one output data (Tensor<T>) where the function f(x) = x^exponent,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">produces one output data (Tensor<T>) where the function f(x) = x^exponent,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">is applied to the data tensor elementwise.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">is applied to the data tensor elementwise.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>_.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>_.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  First operand, base of the exponent.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  First operand, base of the exponent.</code></td></tr>
    <tr style="1px solid black;"><td><code>9</code></td><td><code>9</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">* **Y** (heterogeneous) - **T**:</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>* **Y** (heterogeneous) - **T<span style="color:#196F3D;">1</span>**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Second operand, power of the exponent.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Second operand, power of the exponent.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Z** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Z** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td><code>22</code></td><td><code>22</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  tensor(float16)</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  tensor(float16)<span style="color:#196F3D;">,</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">23</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">24</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td><code>24</code></td><td><code>26</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Constrain input and output types to float tensors.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Constrain input <span style="color:#196F3D;">X</span><span style="color:#196F3D;"> </span>and output types to float<span style="color:#196F3D;">/</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span> tensors.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">27</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **T1** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">28</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">29</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">30</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">31</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(int16),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">32</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">33</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">34</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">35</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(uint16),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">36</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">37</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(uint64),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">38</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">39</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  ):</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">40</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Constrain input Y types to float/int tensors.</code></td></tr>
    </table>

.. _l-onnx-op-pow-7:

Pow - 7
=======

**Version**

* **name**: `Pow (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pow>`_
* **domain**: **main**
* **since_version**: **7**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 7**.

**Summary**

Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check `Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>`_.

**Inputs**

* **X** (heterogeneous) - **T**:
  First operand, base of the exponent.
* **Y** (heterogeneous) - **T**:
  Second operand, power of the exponent.

**Outputs**

* **Z** (heterogeneous) - **T**:
  Output tensor.

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Pow takes input data (Tensor<T>) and exponent Tensor, and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Pow takes input data (Tensor<T>) and exponent Tensor, and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">produces one output data (Tensor<T>) where the function f(x) = x^exponent,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">produces one output data (Tensor<T>) where the function f(x) = x^exponent,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">is applied to the data tensor elementwise.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">is applied to the data tensor elementwise.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">3</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;"></code></td><td></td></tr>
    <tr style="1px solid black;"><td><code>4</code></td><td><code>3</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">I</span><span style="color:#BA4A00;">f</span> <span style="color:#BA4A00;">n</span>e<span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">e</span>ssa<span style="color:#BA4A00;">r</span>y<span style="color:#BA4A00;"> </span>t<span style="color:#BA4A00;">h</span>e r<span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">-</span><span style="color:#BA4A00;">h</span>a<span style="color:#BA4A00;">n</span>d<span style="color:#BA4A00;">-</span>si<span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">e</span> <span style="color:#BA4A00;">a</span>r<span style="color:#BA4A00;">g</span><span style="color:#BA4A00;">u</span>me<span style="color:#BA4A00;">n</span>t<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">w</span>ill <span style="color:#BA4A00;">b</span>e <span style="color:#BA4A00;">b</span>roadcast<span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">d</span> to<span style="color:#BA4A00;"> </span>matc<span style="color:#BA4A00;">h</span><span style="color:#BA4A00;"> </span>t<span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">T</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span> <span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span>e<span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span>s<span style="color:#196F3D;">u</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">t</span>s<span style="color:#196F3D;"> </span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span>a<span style="color:#196F3D;">l</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">N</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">p</span>y<span style="color:#196F3D;">-</span><span style="color:#196F3D;">s</span>t<span style="color:#196F3D;">y</span><span style="color:#196F3D;">l</span>e<span style="color:#196F3D;">)</span> <span style="color:#196F3D;">b</span>r<span style="color:#196F3D;">o</span>ad<span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span>s<span style="color:#196F3D;">t</span>i<span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">;</span> <span style="color:#196F3D;">f</span><span style="color:#196F3D;">o</span>r<span style="color:#196F3D;"> </span>m<span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span>e<span style="color:#196F3D;"> </span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span>t<span style="color:#196F3D;">a</span>il<span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">p</span>l<span style="color:#196F3D;">e</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span> <span style="color:#196F3D;">c</span><span style="color:#196F3D;">h</span>e<span style="color:#196F3D;">c</span><span style="color:#196F3D;">k</span> <span style="color:#196F3D;">B</span>roadcast<span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span> <span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">O</span><span style="color:#196F3D;">N</span><span style="color:#196F3D;">N</span><span style="color:#196F3D;">X</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"><</span><span style="color:#196F3D;">h</span>t<span style="color:#196F3D;">t</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">:</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">c</span>om<span style="color:#196F3D;">/</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">m</span>a<span style="color:#196F3D;">s</span>t<span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">o</span>c<span style="color:#196F3D;">s</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">B</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">s</span>t<span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">></span><span style="color:#196F3D;">_</span><span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">4</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td><code>5</code></td><td><code>5</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">a</span>p<span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">-</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">-</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">g</span>u<span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span>t<span style="color:#BA4A00;">.</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">W</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">a</span>s<span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">d</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">I</span><span style="color:#196F3D;">n</span>puts<span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">6</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td><code>6</code></td><td><code>7</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span> <span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">n</span> e<span style="color:#BA4A00;">i</span>t<span style="color:#BA4A00;">h</span>er<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span>o<span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span>e<span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">e</span>n<span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">z</span>e<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">1</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">(</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">l</span>u<span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;"> </span>s<span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">r</span> <span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span> <span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">y</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">*</span> <span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">X</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span> <span style="color:#196F3D;">(</span><span style="color:#196F3D;">h</span>etero<span style="color:#196F3D;">g</span>ene<span style="color:#196F3D;">o</span>us<span style="color:#196F3D;">)</span> <span style="color:#196F3D;">-</span> <span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">T</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">:</span></code></td></tr>
    <tr style="1px solid black;"><td><code>7</code></td><td><code>8</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span> <span style="color:#BA4A00;">w</span>it<span style="color:#BA4A00;">h</span> ran<span style="color:#BA4A00;">k</span> <span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">q</span><span style="color:#BA4A00;">u</span>a<span style="color:#BA4A00;">l</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;"> </span>s<span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">l</span>e<span style="color:#BA4A00;">r</span> <span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">n</span> the <span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span>e<span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">s</span>o<span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">)</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">v</span><span style="color:#BA4A00;">i</span>n<span style="color:#BA4A00;">g</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span>t<span style="color:#BA4A00;">s</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code> <span style="color:#196F3D;"> </span><span style="color:#196F3D;">F</span>i<span style="color:#196F3D;">r</span><span style="color:#196F3D;">s</span>t <span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span>ran<span style="color:#196F3D;">d</span><span style="color:#196F3D;">,</span> <span style="color:#196F3D;">b</span>ase <span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span> the e<span style="color:#196F3D;">x</span><span style="color:#196F3D;">p</span>on<span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span>t<span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td><code>8</code></td><td><code>9</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">e</span> <span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">s</span> <span style="color:#BA4A00;">a</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">s</span>et<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span>e<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">i</span>r<span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span>en<span style="color:#BA4A00;">s</span>o<span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">'</span>s <span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">.</span> T<span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">*</span> <span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">Y</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span> <span style="color:#196F3D;">(</span><span style="color:#196F3D;">h</span>eter<span style="color:#196F3D;">o</span><span style="color:#196F3D;">g</span>en<span style="color:#196F3D;">e</span>o<span style="color:#196F3D;">u</span>s<span style="color:#196F3D;">)</span> <span style="color:#196F3D;">-</span> <span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span>T<span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">:</span></code></td></tr>
    <tr style="1px solid black;"><td><code>9</code></td><td><code>10</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">y</span> e<span style="color:#BA4A00;">q</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span> <span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">a</span>pe <span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span>pe<span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">i</span>f<span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">d</span> <span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">y</span><span style="color:#BA4A00;"> </span>the <span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">m</span>e<span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">"</span><span style="color:#BA4A00;">a</span>x<span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">"</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span>n<span style="color:#BA4A00;">d</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span>n<span style="color:#BA4A00;">o</span>t<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">,</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code> <span style="color:#196F3D;"> </span><span style="color:#196F3D;">S</span>e<span style="color:#196F3D;">c</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span> <span style="color:#196F3D;">o</span>pe<span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">,</span> p<span style="color:#196F3D;">o</span><span style="color:#196F3D;">w</span>e<span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span>f the ex<span style="color:#196F3D;">p</span><span style="color:#196F3D;">o</span>n<span style="color:#196F3D;">e</span>nt<span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">10</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">suffix matching is assumed. 1-dim expansion doesn't work yet.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">12</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">For example, the following tensor shapes are supported (with broadcast=1):</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">13</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;"></code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">14</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar tensor</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">15</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  shape(A) = (2, 3, 4, 5), shape(B) = (1, 1), i.e. B is an 1-element tensor</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">16</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  shape(A) = (2, 3, 4, 5), shape(B) = (5,)</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">17</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">18</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">19</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">20</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;"></code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">21</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">Attribute broadcast=1 needs to be passed to enable broadcasting.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">22</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;"></code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">23</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">**Attributes**</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">24</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;"></code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">25</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">* **axis**:</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">26</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  If set, defines the broadcast dimensions. See doc for details.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">27</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">* **broadcast**:</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">28</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  Pass 1 to enable broadcasting Default value is 0.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">29</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;"></code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">30</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">**Inputs**</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">31</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;"></code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">32</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">* **X** (heterogeneous) - **T**:</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">33</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  Input tensor of any shape, base of the exponent.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">34</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">* **Y** (heterogeneous) - **T**:</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">35</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  Input tensor of any shape broadcastable to X shape, the exponent</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">36</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  component.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">37</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;"></code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Z** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Z** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td><code>41</code></td><td><code>15</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Output tensor<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">(</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">z</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">X</span><span style="color:#BA4A00;">)</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Output tensor<span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    </table>

.. _l-onnx-op-pow-1:

Pow - 1
=======

**Version**

* **name**: `Pow (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pow>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1**.

**Summary**

Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.

If necessary the right-hand-side argument will be broadcasted to match the
shape of left-hand-side argument. When broadcasting is specified, the second
tensor can either be of element size 1 (including a scalar tensor and any
tensor with rank equal to or smaller than the first tensor), or having its
shape as a contiguous subset of the first tensor's shape. The starting of the
mutually equal shape is specified by the argument "axis", and if it is not set,
suffix matching is assumed. 1-dim expansion doesn't work yet.

For example, the following tensor shapes are supported (with broadcast=1):

  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (1, 1), i.e. B is an 1-element tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

Attribute `broadcast=1` needs to be passed to enable broadcasting.

**Attributes**

* **axis**:
  If set, defines the broadcast dimensions. See doc for details.
* **broadcast**:
  Pass 1 to enable broadcasting Default value is ``0``.

**Inputs**

* **X** (heterogeneous) - **T**:
  Input tensor of any shape, base of the exponent.
* **Y** (heterogeneous) - **T**:
  Input tensor of any shape broadcastable to X shape, the exponent
  component.

**Outputs**

* **Z** (heterogeneous) - **T**:
  Output tensor (same size as X)

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
