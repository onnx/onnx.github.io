
.. _l-onnx-doc-QuantizeLinear:

==============
QuantizeLinear
==============

.. contents::
    :local:


.. _l-onnx-op-quantizelinear-13:

QuantizeLinear - 13
===================

**Version**

* **name**: `QuantizeLinear (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#QuantizeLinear>`_
* **domain**: **main**
* **since_version**: **13**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 13**.

**Summary**

The linear quantization operator. It consumes a high precision tensor, a scale, and a zero point to compute the low precision / quantized tensor.
The scale factor and zero point must have same shape, and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point).
For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.

**Attributes**

* **axis**:
  (Optional) The axis of the quantization dimension of the input
  tensor. Ignored for per-tensor quantization. Negative value means
  counting dimensions from the back. Accepted range is [-r, r-1] where
  r = rank(input). Default value is ``1``.

**Inputs**

Between 2 and 3 inputs.

* **x** (heterogeneous) - **T1**:
  N-D full precision Input tensor to be quantized.
* **y_scale** (heterogeneous) - **tensor(float)**:
  Scale for doing quantization to get 'y'. It can be a scalar, which
  means per-tensor/layer quantization, or a 1-D Tensor for per-axis
  quantization.
* **y_zero_point** (optional, heterogeneous) - **T2**:
  Zero point for doing quantization to get 'y'. Shape must match
  y_scale. Default is uint8 with zero point of 0 if it's not
  specified.

**Outputs**

* **y** (heterogeneous) - **T2**:
  N-D quantized output tensor. It has same shape as input 'x'.

**Type Constraints**

* **T1** in (
  tensor(float),
  tensor(int32)
  ):
  Constrain 'x' to float or int32 tensor.
* **T2** in (
  tensor(int8),
  tensor(uint8)
  ):
  Constrain 'y_zero_point' and 'y' to 8-bit integer tensor.

**Examples**

**axis**

::

    node = onnx.helper.make_node('QuantizeLinear',
                                 inputs=['x', 'y_scale', 'y_zero_point'],
                                 outputs=['y'],)

    x = np.array([[[[-162, 10],
                    [-100, 232],
                    [-20, -50]],

                   [[-76, 0],
                    [0, 252],
                    [32, -44]],

                   [[245, -485],
                    [-960, -270],
                    [-375, -470]], ], ], dtype=np.float32)
    y_scale = np.array([2, 4, 5], dtype=np.float32)
    y_zero_point = np.array([84, 24, 196], dtype=np.uint8)
    y = (x / y_scale.reshape(1, 3, 1, 1) + y_zero_point.reshape(1, 3, 1, 1)).astype(np.uint8)

    expect(node, inputs=[x, y_scale, y_zero_point], outputs=[y],
           name='test_quantizelinear_axis')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td><code>0</code></td><td><code>0</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">The linear <span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">-</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">/</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">y</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;"> </span>quantization operator. It consumes a high precision tensor, a scale, a zero point to compute the low precision / quantized tensor.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>The linear quantization operator. It consumes a high precision tensor, a scale, a<span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span> <span style="color:#196F3D;">a</span><span style="color:#196F3D;"> </span>zero point to compute the low precision / quantized tensor.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">1</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">The scale factor and zero point must have same shape, and can be either a scalar for per-tensor / per layer quantization, or a 1-D tensor for per-axis quantization.</code></td></tr>
    <tr style="1px solid black;"><td><code>1</code></td><td><code>2</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">The quantization formula is y = saturate ((x / y_scale) + y_zero_point).<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">F</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">[</span><span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">2</span><span style="color:#BA4A00;">5</span><span style="color:#BA4A00;">5</span><span style="color:#BA4A00;">]</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">'</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">8</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">[</span><span style="color:#BA4A00;">-</span><span style="color:#BA4A00;">1</span><span style="color:#BA4A00;">2</span><span style="color:#BA4A00;">8</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">1</span><span style="color:#BA4A00;">2</span><span style="color:#BA4A00;">7</span><span style="color:#BA4A00;">]</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">'</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">8</span><span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>The quantization formula is y = saturate ((x / y_scale) + y_zero_point).</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">3</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">For (x / y_scale), it's rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">For (x / y_scale), it's rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">6</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">7</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">8</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **axis**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">9</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  (Optional) The axis of the quantization dimension of the input</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">10</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor. Ignored for per-tensor quantization. Negative value means</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">11</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  counting dimensions from the back. Accepted range is [-r, r-1] where</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">12</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  r = rank(input). Default value is 1.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">13</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 3 inputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 3 inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **x** (heterogeneous) - **T1**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **x** (heterogeneous) - **T1**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  N-D full precision Input tensor to be quantized.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  N-D full precision Input tensor to be quantized.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **y_scale** (heterogeneous) - **tensor(float)**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **y_scale** (heterogeneous) - **tensor(float)**:</code></td></tr>
    <tr style="1px solid black;"><td><code>11</code></td><td><code>21</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Scale for doing quantization to get 'y'. It<span style="color:#BA4A00;">'</span><span style="color:#BA4A00;">s</span> a scalar, which<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">s</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Scale for doing quantization to get 'y'. It <span style="color:#196F3D;">c</span>a<span style="color:#196F3D;">n</span> <span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;"> </span>scalar, which</code></td></tr>
    <tr style="1px solid black;"><td><code>12</code></td><td><code>22</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  a per-tensor/layer quantization<span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span>a<span style="color:#196F3D;">n</span><span style="color:#196F3D;">s</span> per-tensor/layer quantization<span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">D</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">T</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">23</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  quantization.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **y_zero_point** (optional, heterogeneous) - **T2**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **y_zero_point** (optional, heterogeneous) - **T2**:</code></td></tr>
    <tr style="1px solid black;"><td><code>14</code></td><td><code>25</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Zero point for doing quantization to get 'y'. <span style="color:#BA4A00;">I</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">'</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span>a s<span style="color:#BA4A00;">c</span>a<span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">w</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">i</span>ch</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Zero point for doing quantization to get 'y'. <span style="color:#196F3D;">S</span><span style="color:#196F3D;">h</span>a<span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span> <span style="color:#196F3D;">m</span><span style="color:#196F3D;">u</span>s<span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">m</span>a<span style="color:#196F3D;">t</span>ch</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">15</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  means a per-tensor/layer quantization. Default value is uint8 typed</code></td><td></td></tr>
    <tr style="1px solid black;"><td><code>16</code></td><td><code>26</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  0 if it's not<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">y</span><span style="color:#196F3D;">_</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">D</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">8</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">w</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">z</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span>0 if it's not</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">27</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  specified.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **y** (heterogeneous) - **T2**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **y** (heterogeneous) - **T2**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  N-D quantized output tensor. It has same shape as input 'x'.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  N-D quantized output tensor. It has same shape as input 'x'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain 'x' to float or int32 tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain 'x' to float or int32 tensor.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T2** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T2** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain 'y_zero_point' and 'y' to 8-bit integer tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain 'y_zero_point' and 'y' to 8-bit integer tensor.</code></td></tr>
    </table>

.. _l-onnx-op-quantizelinear-10:

QuantizeLinear - 10
===================

**Version**

* **name**: `QuantizeLinear (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#QuantizeLinear>`_
* **domain**: **main**
* **since_version**: **10**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 10**.

**Summary**

The linear per-tensor/layer quantization operator. It consumes a high precision tensor, a scale, a zero point to compute the low precision / quantized tensor.
The quantization formula is y = saturate ((x / y_scale) + y_zero_point). For saturation, it saturates to [0, 255] if it's uint8, or [-128, 127] if it's int8.
For (x / y_scale), it's rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details. 'y_zero_point' and 'y' must have same type.

**Inputs**

Between 2 and 3 inputs.

* **x** (heterogeneous) - **T1**:
  N-D full precision Input tensor to be quantized.
* **y_scale** (heterogeneous) - **tensor(float)**:
  Scale for doing quantization to get 'y'. It's a scalar, which means
  a per-tensor/layer quantization.
* **y_zero_point** (optional, heterogeneous) - **T2**:
  Zero point for doing quantization to get 'y'. It's a scalar, which
  means a per-tensor/layer quantization. Default value is uint8 typed
  0 if it's not specified.

**Outputs**

* **y** (heterogeneous) - **T2**:
  N-D quantized output tensor. It has same shape as input 'x'.

**Type Constraints**

* **T1** in (
  tensor(float),
  tensor(int32)
  ):
  Constrain 'x' to float or int32 tensor.
* **T2** in (
  tensor(int8),
  tensor(uint8)
  ):
  Constrain 'y_zero_point' and 'y' to 8-bit integer tensor.
