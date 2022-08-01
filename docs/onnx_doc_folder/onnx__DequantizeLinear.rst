
.. _l-onnx-doc-DequantizeLinear:

================
DequantizeLinear
================

.. contents::
    :local:


.. _l-onnx-op-dequantizelinear-13:

DequantizeLinear - 13
=====================

**Version**

* **name**: `DequantizeLinear (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#DequantizeLinear>`_
* **domain**: **main**
* **since_version**: **13**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 13**.

**Summary**

The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute the full precision tensor.
The dequantization formula is y = (x - x_zero_point) * x_scale. 'x_scale' and 'x_zero_point' must have same shape, and can be either a scalar
for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
'x_zero_point' and 'x' must have same type. 'x' and 'y' must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).

**Attributes**

* **axis**:
  (Optional) The axis of the dequantizing dimension of the input
  tensor. Ignored for per-tensor quantization. Negative value means
  counting dimensions from the back. Accepted range is [-r, r-1] where
  r = rank(input). Default value is ``1``.

**Inputs**

Between 2 and 3 inputs.

* **x** (heterogeneous) - **T**:
  N-D quantized input tensor to be de-quantized.
* **x_scale** (heterogeneous) - **tensor(float)**:
  Scale for input 'x'. It can be a scalar, which means a per-
  tensor/layer dequantization, or a 1-D tensor for per-axis
  dequantization.
* **x_zero_point** (optional, heterogeneous) - **T**:
  Zero point for input 'x'. Shape must match x_scale. It's optional.
  Zero point is 0 when it's not specified.

**Outputs**

* **y** (heterogeneous) - **tensor(float)**:
  N-D full precision output tensor. It has same shape as input 'x'.

**Type Constraints**

* **T** in (
  tensor(int32),
  tensor(int8),
  tensor(uint8)
  ):
  Constrain 'x_zero_point' and 'x' to 8-bit/32-bit integer tensor.

**Examples**

**axis**

::

    node = onnx.helper.make_node('DequantizeLinear',
                                 inputs=['x', 'x_scale', 'x_zero_point'],
                                 outputs=['y'],)

    # 1-D tensor zero point and scale of size equal to axis 1 of the input tensor
    x = np.array([[[[3, 89],
                    [34, 200],
                    [74, 59]],

                   [[5, 24],
                    [24, 87],
                    [32, 13]],

                   [[245, 99],
                    [4, 142],
                    [121, 102]], ], ], dtype=np.uint8)
    x_scale = np.array([2, 4, 5], dtype=np.float32)
    x_zero_point = np.array([84, 24, 196], dtype=np.uint8)
    y = (x.astype(np.float32) - x_zero_point.reshape(1, 3, 1, 1).astype(np.float32)) * x_scale.reshape(1, 3, 1, 1)

    expect(node, inputs=[x, x_scale, x_zero_point], outputs=[y],
           name='test_dequantizelinear_axis')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td><code>0</code></td><td><code>0</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">The linear dequantization operator. It consumes a quantized tensor, a scale, a zero point to compute the full precision tensor.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>The linear dequantization operator. It consumes a quantized tensor, a scale, a<span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span> <span style="color:#196F3D;">a</span><span style="color:#196F3D;"> </span>zero point to compute the full precision tensor.</code></td></tr>
    <tr style="1px solid black;"><td><code>1</code></td><td><code>1</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">The dequantization formula is y = (x - x_zero_point) * x_scale. 'x_scale' and 'x_zero_point' a<span style="color:#BA4A00;">r</span>e b<span style="color:#BA4A00;">o</span>th scalar<span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>The dequantization formula is y = (x - x_zero_point) * x_scale. 'x_scale' and 'x_zero_point' <span style="color:#196F3D;">m</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">h</span>a<span style="color:#196F3D;">v</span>e <span style="color:#196F3D;">s</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span>b<span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">i</span>th<span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span> <span style="color:#196F3D;">a</span><span style="color:#196F3D;"> </span>scalar</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">2</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">'x_zero_point' and 'x' must have same type. 'x' and 'y' must have same shape. In the case of dequantizing int32,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">'x_zero_point' and 'x' must have same type. 'x' and 'y' must have same shape. In the case of dequantizing int32,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">there's no zero point (zero point is supposed to be 0).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">there's no zero point (zero point is supposed to be 0).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">6</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">7</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">8</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **axis**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">9</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  (Optional) The axis of the dequantizing dimension of the input</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">10</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor. Ignored for per-tensor quantization. Negative value means</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">11</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  counting dimensions from the back. Accepted range is [-r, r-1] where</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">12</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  r = rank(input). Default value is 1.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">13</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 3 inputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 3 inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **x** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **x** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  N-D quantized input tensor to be de-quantized.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  N-D quantized input tensor to be de-quantized.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **x_scale** (heterogeneous) - **tensor(float)**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **x_scale** (heterogeneous) - **tensor(float)**:</code></td></tr>
    <tr style="1px solid black;"><td><code>12</code></td><td><code>21</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Scale for input 'x'. It<span style="color:#BA4A00;">'</span><span style="color:#BA4A00;">s</span> a scalar, which means a per-<span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">/</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">y</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">r</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Scale for input 'x'. It <span style="color:#196F3D;">c</span>a<span style="color:#196F3D;">n</span> <span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;"> </span>scalar, which means a per-</code></td></tr>
    <tr style="1px solid black;"><td><code>13</code></td><td><code>22</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  quantization<span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span>quantization<span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">D</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">23</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  dequantization.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **x_zero_point** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **x_zero_point** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td><code>15</code></td><td><code>25</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Zero point for input 'x'. <span style="color:#BA4A00;">I</span>t<span style="color:#BA4A00;">'</span><span style="color:#BA4A00;">s</span> a scal<span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">,</span> <span style="color:#BA4A00;">w</span><span style="color:#BA4A00;">h</span>i<span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">a</span>n<span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span>a<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">-</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Zero point for input 'x'. <span style="color:#196F3D;">S</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">s</span>t <span style="color:#196F3D;">m</span>a<span style="color:#196F3D;">t</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">h</span> <span style="color:#196F3D;">x</span><span style="color:#196F3D;">_</span>scal<span style="color:#196F3D;">e</span><span style="color:#196F3D;">.</span> <span style="color:#196F3D;">I</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">'</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span>i<span style="color:#196F3D;">o</span>na<span style="color:#196F3D;">l</span><span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">16</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  tensor/layer quantization. It's optional. 0 is the default value</code></td><td></td></tr>
    <tr style="1px solid black;"><td><code>17</code></td><td><code>26</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  when it's not specified.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">Z</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">0</span><span style="color:#196F3D;"> </span>when it's not specified.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **y** (heterogeneous) - **tensor(float)**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **y** (heterogeneous) - **tensor(float)**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  N-D full precision output tensor. It has same shape as input 'x'.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  N-D full precision output tensor. It has same shape as input 'x'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain 'x_zero_point' and 'x' to 8-bit/32-bit integer tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain 'x_zero_point' and 'x' to 8-bit/32-bit integer tensor.</code></td></tr>
    </table>

.. _l-onnx-op-dequantizelinear-10:

DequantizeLinear - 10
=====================

**Version**

* **name**: `DequantizeLinear (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#DequantizeLinear>`_
* **domain**: **main**
* **since_version**: **10**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 10**.

**Summary**

The linear dequantization operator. It consumes a quantized tensor, a scale, a zero point to compute the full precision tensor.
The dequantization formula is y = (x - x_zero_point) * x_scale. 'x_scale' and 'x_zero_point' are both scalars.
'x_zero_point' and 'x' must have same type. 'x' and 'y' must have same shape. In the case of dequantizing int32,
there's no zero point (zero point is supposed to be 0).

**Inputs**

Between 2 and 3 inputs.

* **x** (heterogeneous) - **T**:
  N-D quantized input tensor to be de-quantized.
* **x_scale** (heterogeneous) - **tensor(float)**:
  Scale for input 'x'. It's a scalar, which means a per-tensor/layer
  quantization.
* **x_zero_point** (optional, heterogeneous) - **T**:
  Zero point for input 'x'. It's a scalar, which means a per-
  tensor/layer quantization. It's optional. 0 is the default value
  when it's not specified.

**Outputs**

* **y** (heterogeneous) - **tensor(float)**:
  N-D full precision output tensor. It has same shape as input 'x'.

**Type Constraints**

* **T** in (
  tensor(int32),
  tensor(int8),
  tensor(uint8)
  ):
  Constrain 'x_zero_point' and 'x' to 8-bit/32-bit integer tensor.
