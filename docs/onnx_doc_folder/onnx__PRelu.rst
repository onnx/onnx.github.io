
.. _l-onnx-doc-PRelu:

=====
PRelu
=====

.. contents::
    :local:


.. _l-onnx-op-prelu-16:

PRelu - 16
==========

**Version**

* **name**: `PRelu (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#PRelu>`_
* **domain**: **main**
* **since_version**: **16**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 16**.

**Summary**

PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.

**History**
- Version 16 adds bfloat16 to the types allowed.
This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check `Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>`_.

**Inputs**

* **X** (heterogeneous) - **T**:
  Input tensor
* **slope** (heterogeneous) - **T**:
  Slope tensor. The shape of slope can be smaller then first input X;
  if so, its shape must be unidirectional broadcastable to X

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output tensor (same size as X)

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
  Constrain input and output types to float/int tensors.

**Examples**

**prelu_broadcast**

::

    node = onnx.helper.make_node(
        'PRelu',
        inputs=['x', 'slope'],
        outputs=['y'],
    )

    x = np.random.randn(3, 4, 5).astype(np.float32)
    slope = np.random.randn(5).astype(np.float32)
    y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope

    expect(node, inputs=[x, slope], outputs=[y],
           name='test_prelu_broadcast')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">output data (Tensor<T>) where the function f(x) = slope * x for x < 0,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">output data (Tensor<T>) where the function f(x) = slope * x for x < 0,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">f(x) = x for x >= 0., is applied to the data tensor elementwise.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">f(x) = x for x >= 0., is applied to the data tensor elementwise.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">3</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">4</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">**History**</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">5</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">- Version 16 adds bfloat16 to the types allowed.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>_.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>_.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **slope** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **slope** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Slope tensor. The shape of slope can be smaller then first input X;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Slope tensor. The shape of slope can be smaller then first input X;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  if so, its shape must be unidirectional broadcastable to X</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  if so, its shape must be unidirectional broadcastable to X</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor (same size as X)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor (same size as X)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">24</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(bfloat16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float/int tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float/int tensors.</code></td></tr>
    </table>

.. _l-onnx-op-prelu-9:

PRelu - 9
=========

**Version**

* **name**: `PRelu (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#PRelu>`_
* **domain**: **main**
* **since_version**: **9**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 9**.

**Summary**

PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check `Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>`_.

**Inputs**

* **X** (heterogeneous) - **T**:
  Input tensor
* **slope** (heterogeneous) - **T**:
  Slope tensor. The shape of slope can be smaller then first input X;
  if so, its shape must be unidirectional broadcastable to X

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output tensor (same size as X)

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
  Constrain input and output types to float/int tensors.

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">output data (Tensor<T>) where the function f(x) = slope * x for x < 0,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">output data (Tensor<T>) where the function f(x) = slope * x for x < 0,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">f(x) = x for x >= 0., is applied to the data tensor elementwise.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">f(x) = x for x >= 0., is applied to the data tensor elementwise.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>_.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>_.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **slope** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **slope** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Slope tensor. The shape of slope can be smaller then first input X;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Slope tensor. The shape of slope can be smaller then first input X;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  if so, its shape must be unidirectional broadcastable to X</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  if so, its shape must be unidirectional broadcastable to X</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor (same size as X)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor (same size as X)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td><code>23</code></td><td><code>23</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  tensor(float16)</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  tensor(float16)<span style="color:#196F3D;">,</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">24</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">25</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">26</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">27</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(uint64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td><code>25</code></td><td><code>29</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Constrain input and output types to float tensors.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Constrain input and output types to float<span style="color:#196F3D;">/</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span> tensors.</code></td></tr>
    </table>

.. _l-onnx-op-prelu-7:

PRelu - 7
=========

**Version**

* **name**: `PRelu (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#PRelu>`_
* **domain**: **main**
* **since_version**: **7**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 7**.

**Summary**

PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check `Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>`_.

**Inputs**

* **X** (heterogeneous) - **T**:
  Input tensor
* **slope** (heterogeneous) - **T**:
  Slope tensor. The shape of slope can be smaller then first input X;
  if so, its shape must be unidirectional broadcastable to X

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output tensor (same size as X)

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
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">output data (Tensor<T>) where the function f(x) = slope * x for x < 0,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">output data (Tensor<T>) where the function f(x) = slope * x for x < 0,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">f(x) = x for x >= 0., is applied to the data tensor elementwise.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">f(x) = x for x >= 0., is applied to the data tensor elementwise.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">3</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>_.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **slope** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **slope** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td><code>9</code></td><td><code>10</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Slope tensor. <span style="color:#BA4A00;">I</span><span style="color:#BA4A00;">f</span> <span style="color:#BA4A00;">S</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">o</span>pe <span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span>of s<span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">z</span>e <span style="color:#BA4A00;">1</span><span style="color:#BA4A00;">,</span> <span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span>e <span style="color:#BA4A00;">v</span>al<span style="color:#BA4A00;">u</span>e <span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span>h<span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">r</span>e<span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">c</span>r<span style="color:#BA4A00;">o</span>s<span style="color:#BA4A00;">s</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Slope tensor. <span style="color:#196F3D;">T</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span> <span style="color:#196F3D;">s</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span>pe of s<span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span>e <span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span> <span style="color:#196F3D;">b</span>e <span style="color:#196F3D;">s</span><span style="color:#196F3D;">m</span>al<span style="color:#196F3D;">l</span>e<span style="color:#196F3D;">r</span> <span style="color:#196F3D;">t</span>he<span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">i</span>rs<span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">X</span><span style="color:#196F3D;">;</span></code></td></tr>
    <tr style="1px solid black;"><td><code>10</code></td><td><code>11</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  <span style="color:#BA4A00;">d</span>if<span style="color:#BA4A00;">f</span>eren<span style="color:#BA4A00;">t</span> c<span style="color:#BA4A00;">h</span>a<span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">n</span>e<span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">s</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  if<span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">p</span>e<span style="color:#196F3D;"> </span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span>re<span style="color:#196F3D;">c</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span>n<span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span> <span style="color:#196F3D;">b</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">d</span>ca<span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">l</span>e<span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">X</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td><code>15</code></td><td><code>16</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Output tensor</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Output tensor<span style="color:#196F3D;"> </span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">z</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">X</span><span style="color:#196F3D;">)</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    </table>

.. _l-onnx-op-prelu-6:

PRelu - 6
=========

**Version**

* **name**: `PRelu (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#PRelu>`_
* **domain**: **main**
* **since_version**: **6**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 6**.

**Summary**

PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.

**Inputs**

* **X** (heterogeneous) - **T**:
  Input tensor
* **slope** (heterogeneous) - **T**:
  Slope tensor. If `Slope` is of size 1, the value is sharedacross
  different channels

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output tensor

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
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">output data (Tensor<T>) where the function f(x) = slope * x for x < 0,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">output data (Tensor<T>) where the function f(x) = slope * x for x < 0,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">f(x) = x for x >= 0., is applied to the data tensor elementwise.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">f(x) = x for x >= 0., is applied to the data tensor elementwise.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">4</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">**Attributes**</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">5</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;"></code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">6</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">* **consumed_inputs**:</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">7</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  legacy optimization attribute.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">8</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;"></code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **slope** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **slope** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Slope tensor. If Slope is of size 1, the value is sharedacross</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Slope tensor. If Slope is of size 1, the value is sharedacross</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  different channels</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  different channels</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    </table>

.. _l-onnx-op-prelu-1:

PRelu - 1
=========

**Version**

* **name**: `PRelu (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#PRelu>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: False

This version of the operator has been available
**since version 1**.

**Summary**

PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.

**Attributes**

* **consumed_inputs**:
  legacy optimization attribute.

**Inputs**

* **X** (heterogeneous) - **T**:
  Input tensor
* **slope** (heterogeneous) - **T**:
  Slope tensor. If `Slope` is of size 1, the value is sharedacross
  different channels

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output tensor

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
