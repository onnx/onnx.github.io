
.. _l-onnx-doc-Softmax:

=======
Softmax
=======

.. contents::
    :local:


.. _l-onnx-op-softmax-13:

Softmax - 13
============

**Version**

* **name**: `Softmax (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax>`_
* **domain**: **main**
* **since_version**: **13**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 13**.

**Summary**

The operator computes the normalized exponential values for the given input:

 Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)

The "axis" attribute indicates the dimension along which Softmax
will be performed. The output tensor has the same shape
and contains the Softmax values of the corresponding input.

**Attributes**

* **axis**:
   Describes the dimension Softmax will be performed on. Negative
  value means counting dimensions from the back. Accepted range is
  [-r, r-1] where r = rank(input). Default value is ``-1``.

**Inputs**

* **input** (heterogeneous) - **T**:
  The input tensor of rank >= axis.

**Outputs**

* **output** (heterogeneous) - **T**:
  The output values with the same shape as the input tensor.

**Type Constraints**

* **T** in (
  tensor(bfloat16),
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.

**Examples**

**softmax_axis**

::

    x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]
                 ).astype(np.float32)
    # expected output
    # [[0.032058604 0.08714432  0.23688284  0.6439143  ]
    # [0.032058604 0.08714432  0.23688284  0.6439143  ]]
    y = softmax(x)

    node = onnx.helper.make_node(
        'Softmax',
        inputs=['x'],
        outputs=['y'],
    )
    expect(node, inputs=[x], outputs=[y],
           name='test_softmax_large_number')

    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    node = onnx.helper.make_node(
        'Softmax',
        inputs=['x'],
        outputs=['y'],
        axis=0,
    )
    y = softmax(x, axis=0)
    expect(node, inputs=[x], outputs=[y],
           name='test_softmax_axis_0')

    node = onnx.helper.make_node(
        'Softmax',
        inputs=['x'],
        outputs=['y'],
        axis=1,
    )
    y = softmax(x, axis=1)
    expect(node, inputs=[x], outputs=[y],
           name='test_softmax_axis_1')

    node = onnx.helper.make_node(
        'Softmax',
        inputs=['x'],
        outputs=['y'],
        axis=2,
    )
    y = softmax(x, axis=2)
    expect(node, inputs=[x], outputs=[y],
           name='test_softmax_axis_2')

    node = onnx.helper.make_node(
        'Softmax',
        inputs=['x'],
        outputs=['y'],
        axis=-1,
    )
    y = softmax(x, axis=-1)
    expect(node, inputs=[x], outputs=[y],
           name='test_softmax_negative_axis')

    # default axis is -1
    node = onnx.helper.make_node(
        'Softmax',
        inputs=['x'],
        outputs=['y'],
    )
    expect(node, inputs=[x], outputs=[y],
           name='test_softmax_default_axis')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td><code>0</code></td><td><code>0</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">The operator computes the <span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">x</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">(</span>normalized exponential<span style="color:#BA4A00;">)</span> values for e<span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">h</span> <span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">y</span>e<span style="color:#BA4A00;">r</span> in<span style="color:#BA4A00;"> </span>t<span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">h</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>The operator computes the normalized exponential values for <span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span>e <span style="color:#196F3D;">g</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">v</span>e<span style="color:#196F3D;">n</span> in<span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span>t<span style="color:#196F3D;">:</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">1</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td><code>1</code></td><td><code>2</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"> of<span style="color:#BA4A00;"> </span>t<span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span> <span style="color:#BA4A00;">g</span>i<span style="color:#BA4A00;">v</span><span style="color:#BA4A00;">e</span>n input<span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code> <span style="color:#196F3D;">S</span>oft<span style="color:#196F3D;">m</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">,</span> <span style="color:#196F3D;">a</span><span style="color:#196F3D;">x</span>i<span style="color:#196F3D;">s</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">=</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">E</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">i</span>n<span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">)</span> <span style="color:#196F3D;">/</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">R</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">S</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">E</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">(</span>input<span style="color:#196F3D;">)</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">=</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">k</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">=</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">)</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">3</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">The input does not need to explicitly be a 2D vector; rather, it will be</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">4</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">coerced into one. For an arbitrary n-dimensional tensor</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">5</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is</code></td><td></td></tr>
    <tr style="1px solid black;"><td><code>6</code></td><td><code>4</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">t</span>he axis <span style="color:#BA4A00;">p</span>r<span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">v</span>i<span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span>t<span style="color:#BA4A00;">h</span>e<span style="color:#BA4A00;">n</span> in<span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">w</span>i<span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span>c<span style="color:#BA4A00;">o</span>e<span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">d</span> <span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span>t<span style="color:#BA4A00;">o</span> <span style="color:#BA4A00;">a</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">2</span><span style="color:#BA4A00;">-</span>dimensional<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span>n<span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span> wit<span style="color:#BA4A00;">h</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">T</span>he <span style="color:#196F3D;">"</span>axis<span style="color:#196F3D;">"</span> <span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">t</span>ri<span style="color:#196F3D;">b</span><span style="color:#196F3D;">u</span>te in<span style="color:#196F3D;">d</span>ic<span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span>e<span style="color:#196F3D;">s</span> t<span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span> dimension<span style="color:#196F3D;"> </span>al<span style="color:#196F3D;">o</span>n<span style="color:#196F3D;">g</span> w<span style="color:#196F3D;">h</span>i<span style="color:#196F3D;">c</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">S</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span>t<span style="color:#196F3D;">m</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">x</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">7</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">8</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">case where axis=1, this means the input tensor will be coerced into a 2D tensor</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">9</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">10</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">11</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">Each of these dimensions must be matched correctly, or else the operator</code></td><td></td></tr>
    <tr style="1px solid black;"><td><code>12</code></td><td><code>5</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">will <span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">w</span> er<span style="color:#BA4A00;">r</span>or<span style="color:#BA4A00;">s</span>. The output tensor has the same shape</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>will <span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span> <span style="color:#196F3D;">p</span>er<span style="color:#196F3D;">f</span>or<span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span>. The output tensor has the same shape</code></td></tr>
    <tr style="1px solid black;"><td><code>13</code></td><td><code>6</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">and contains the <span style="color:#BA4A00;">s</span>oftmax values of the corresponding input.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>and contains the <span style="color:#196F3D;">S</span>oftmax values of the corresponding input.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **axis**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **axis**:</code></td></tr>
    <tr style="1px solid black;"><td><code>18</code></td><td><code>11</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Describes the <span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">x</span>is of<span style="color:#BA4A00;"> </span>t<span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span> i<span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">s</span> <span style="color:#BA4A00;">w</span><span style="color:#BA4A00;">h</span>e<span style="color:#BA4A00;">n</span> <span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">o</span>er<span style="color:#BA4A00;">c</span>ed <span style="color:#BA4A00;">t</span>o <span style="color:#BA4A00;">2</span><span style="color:#BA4A00;">D</span><span style="color:#BA4A00;">;</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">d</span>e<span style="color:#BA4A00;">f</span>a<span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">l</span>t<span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">n</span>e</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;"> </span>Describes the <span style="color:#196F3D;">d</span>i<span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span>s<span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span> <span style="color:#196F3D;">S</span>oft<span style="color:#196F3D;">m</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">x</span> <span style="color:#196F3D;">w</span>i<span style="color:#196F3D;">l</span><span style="color:#196F3D;">l</span> <span style="color:#196F3D;">b</span>e <span style="color:#196F3D;">p</span>er<span style="color:#196F3D;">f</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">m</span>ed o<span style="color:#196F3D;">n</span><span style="color:#196F3D;">.</span> <span style="color:#196F3D;">N</span>e<span style="color:#196F3D;">g</span>at<span style="color:#196F3D;">i</span><span style="color:#196F3D;">v</span>e</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">19</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  because the 0th axis most likely describes the batch_size. Negative</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  value means counting dimensions from the back. Accepted range is</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  value means counting dimensions from the back. Accepted range is</code></td></tr>
    <tr style="1px solid black;"><td><code>21</code></td><td><code>13</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  [-r, r-1] where r = rank(input). Default value is 1.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  [-r, r-1] where r = rank(input). Default value is <span style="color:#196F3D;">-</span>1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **input** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **input** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td><code>26</code></td><td><code>18</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  The input tensor <span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">'</span><span style="color:#BA4A00;">s</span> <span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">e</span>r<span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span>n<span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">o</span> <span style="color:#BA4A00;">a</span> <span style="color:#BA4A00;">2</span><span style="color:#BA4A00;">D</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">m</span>a<span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">i</span>x<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span>i<span style="color:#BA4A00;">z</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">(</span><span style="color:#BA4A00;">N</span><span style="color:#BA4A00;">x</span><span style="color:#BA4A00;">D</span><span style="color:#BA4A00;">)</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span>s</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  The input tensor <span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span> r<span style="color:#196F3D;">a</span>n<span style="color:#196F3D;">k</span> <span style="color:#196F3D;">></span><span style="color:#196F3D;">=</span> axis<span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">27</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  described above.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td><code>32</code></td><td><code>23</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  The output values with the same shape as input tensor<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">(</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  The output values with the same shape as <span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span>input tensor<span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">33</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  size without coercion).</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">28</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(bfloat16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    </table>

.. _l-onnx-op-softmax-11:

Softmax - 11
============

**Version**

* **name**: `Softmax (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

The operator computes the softmax (normalized exponential) values for each layer in the batch
 of the given input.

The input does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
the axis provided, then input will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the input tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
Each of these dimensions must be matched correctly, or else the operator
will throw errors. The output tensor has the same shape
and contains the softmax values of the corresponding input.

**Attributes**

* **axis**:
  Describes the axis of the inputs when coerced to 2D; defaults to one
  because the 0th axis most likely describes the batch_size. Negative
  value means counting dimensions from the back. Accepted range is
  [-r, r-1] where r = rank(input). Default value is ``1``.

**Inputs**

* **input** (heterogeneous) - **T**:
  The input tensor that's coerced into a 2D matrix of size (NxD) as
  described above.

**Outputs**

* **output** (heterogeneous) - **T**:
  The output values with the same shape as input tensor (the original
  size without coercion).

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
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The operator computes the softmax (normalized exponential) values for each layer in the batch</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The operator computes the softmax (normalized exponential) values for each layer in the batch</code></td></tr>
    <tr style="1px solid black;"><td><code>1</code></td><td><code>1</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"> of the given input.<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">T</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">2</span><span style="color:#BA4A00;">-</span><span style="color:#BA4A00;">D</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">(</span><span style="color:#BA4A00;">T</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;"><</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">></span><span style="color:#BA4A00;">)</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">z</span><span style="color:#BA4A00;">e</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code> of the given input.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">2</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">(batch_size x input_feature_dimensions). The output tensor has the same shape</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">3</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">and contains the softmax values of the corresponding input.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td><code>5</code></td><td><code>3</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">I</span>nput does not need to explicitly be a 2D vector; rather, it will be</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">T</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span>nput does not need to explicitly be a 2D vector; rather, it will be</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">coerced into one. For an arbitrary n-dimensional tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">coerced into one. For an arbitrary n-dimensional tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the axis provided, then input will be coerced into a 2-dimensional tensor with</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the axis provided, then input will be coerced into a 2-dimensional tensor with</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">case where axis=1, this means the input tensor will be coerced into a 2D tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">case where axis=1, this means the input tensor will be coerced into a 2D tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Each of these dimensions must be matched correctly, or else the operator</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Each of these dimensions must be matched correctly, or else the operator</code></td></tr>
    <tr style="1px solid black;"><td><code>14</code></td><td><code>12</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">will throw errors.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>will throw errors.<span style="color:#196F3D;"> </span><span style="color:#196F3D;">T</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">13</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">and contains the softmax values of the corresponding input.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **axis**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **axis**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Describes the axis of the inputs when coerced to 2D; defaults to one</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Describes the axis of the inputs when coerced to 2D; defaults to one</code></td></tr>
    <tr style="1px solid black;"><td><code>20</code></td><td><code>19</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  because the 0th axis most likely describes the batch_size <span style="color:#BA4A00;">D</span>e<span style="color:#BA4A00;">f</span>a<span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">l</span>t<span style="color:#BA4A00;"> </span>v<span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">u</span>e<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">1</span><span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  because the 0th axis most likely describes the batch_size<span style="color:#196F3D;">.</span> <span style="color:#196F3D;">N</span>e<span style="color:#196F3D;">g</span>at<span style="color:#196F3D;">i</span>ve</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">20</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  value means counting dimensions from the back. Accepted range is</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">21</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  [-r, r-1] where r = rank(input). Default value is 1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **input** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **input** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input tensor that's coerced into a 2D matrix of size (NxD) as</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input tensor that's coerced into a 2D matrix of size (NxD) as</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  described above.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  described above.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output values with the same shape as input tensor (the original</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output values with the same shape as input tensor (the original</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  size without coercion).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  size without coercion).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    </table>

.. _l-onnx-op-softmax-1:

Softmax - 1
===========

**Version**

* **name**: `Softmax (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1**.

**Summary**

The operator computes the softmax (normalized exponential) values for each layer in the batch
 of the given input. The input is a 2-D tensor (Tensor<float>) of size
(batch_size x input_feature_dimensions). The output tensor has the same shape
and contains the softmax values of the corresponding input.

Input does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
the axis provided, then input will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the input tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
Each of these dimensions must be matched correctly, or else the operator
will throw errors.

**Attributes**

* **axis**:
  Describes the axis of the inputs when coerced to 2D; defaults to one
  because the 0th axis most likely describes the batch_size Default value is ``1``.

**Inputs**

* **input** (heterogeneous) - **T**:
  The input tensor that's coerced into a 2D matrix of size (NxD) as
  described above.

**Outputs**

* **output** (heterogeneous) - **T**:
  The output values with the same shape as input tensor (the original
  size without coercion).

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
