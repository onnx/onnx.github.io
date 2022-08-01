
.. _l-onnx-doc-Selu:

====
Selu
====

.. contents::
    :local:


.. _l-onnx-op-selu-6:

Selu - 6
========

**Version**

* **name**: `Selu (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Selu>`_
* **domain**: **main**
* **since_version**: **6**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 6**.

**Summary**

Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.

**Attributes**

* **alpha**:
  Coefficient of SELU default to 1.67326319217681884765625 (i.e.,
  float32 approximation of 1.6732632423543772848170429916717). Default value is ``1.6732631921768188``.
* **gamma**:
  Coefficient of SELU default to 1.05070102214813232421875 (i.e.,
  float32 approximation of 1.0507009873554804934193349852946). Default value is ``1.0507010221481323``.

**Inputs**

* **X** (heterogeneous) - **T**:
  Input tensor

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

**Examples**

**selu_default**

::

    default_alpha = 1.67326319217681884765625
    default_gamma = 1.05070102214813232421875
    node = onnx.helper.make_node(
        'Selu',
        inputs=['x'],
        outputs=['y'],
    )
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.clip(x, 0, np.inf) * default_gamma + \
        (np.exp(np.clip(x, -np.inf, 0)) - 1) * default_alpha * default_gamma
    expect(node, inputs=[x], outputs=[y],
           name='test_selu_default')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Selu takes one input data (Tensor<T>) and produces one output data</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Selu takes one input data (Tensor<T>) and produces one output data</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">(Tensor<T>) where the scaled exponential linear unit function,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">(Tensor<T>) where the scaled exponential linear unit function,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">y = gamma * (alpha * e^x - alpha) for x <= 0, y = gamma * x for x > 0,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">y = gamma * (alpha * e^x - alpha) for x <= 0, y = gamma * x for x > 0,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">is applied to the tensor elementwise.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">is applied to the tensor elementwise.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **alpha**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **alpha**:</code></td></tr>
    <tr style="1px solid black;"><td><code>8</code></td><td><code>8</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Coefficient of SELU default to 1.6732<span style="color:#BA4A00;">.</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">D</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">v</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">1</span><span style="color:#BA4A00;">.</span>6<span style="color:#BA4A00;">7</span>32<span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">0</span>1125<span style="color:#BA4A00;">3</span><span style="color:#BA4A00;">3</span><span style="color:#BA4A00;">5</span><span style="color:#BA4A00;">7</span>.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Coefficient of SELU default to 1.673263<span style="color:#196F3D;">1</span><span style="color:#196F3D;">9</span>21<span style="color:#196F3D;">7</span><span style="color:#196F3D;">6</span><span style="color:#196F3D;">8</span>1<span style="color:#196F3D;">8</span><span style="color:#196F3D;">8</span><span style="color:#196F3D;">4</span><span style="color:#196F3D;">7</span><span style="color:#196F3D;">6</span><span style="color:#196F3D;">5</span><span style="color:#196F3D;">6</span>25<span style="color:#196F3D;"> </span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">i</span>.<span style="color:#196F3D;">e</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">,</span></code></td></tr>
    <tr style="1px solid black;"><td><code>9</code></td><td><code>9</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">*</span> <span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">c</span>o<span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">u</span>m<span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">_</span>in<span style="color:#BA4A00;">p</span>uts<span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">:</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code> <span style="color:#196F3D;"> </span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">l</span>o<span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">3</span><span style="color:#196F3D;">2</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">i</span>m<span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span>i<span style="color:#196F3D;">o</span>n<span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">6</span><span style="color:#196F3D;">7</span><span style="color:#196F3D;">3</span><span style="color:#196F3D;">2</span><span style="color:#196F3D;">6</span><span style="color:#196F3D;">3</span><span style="color:#196F3D;">2</span><span style="color:#196F3D;">4</span><span style="color:#196F3D;">2</span><span style="color:#196F3D;">3</span><span style="color:#196F3D;">5</span><span style="color:#196F3D;">4</span><span style="color:#196F3D;">3</span><span style="color:#196F3D;">7</span><span style="color:#196F3D;">7</span><span style="color:#196F3D;">2</span><span style="color:#196F3D;">8</span><span style="color:#196F3D;">4</span><span style="color:#196F3D;">8</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">7</span><span style="color:#196F3D;">0</span><span style="color:#196F3D;">4</span><span style="color:#196F3D;">2</span><span style="color:#196F3D;">9</span><span style="color:#196F3D;">9</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">6</span><span style="color:#196F3D;">7</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">7</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">D</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">a</span>u<span style="color:#196F3D;">l</span>t<span style="color:#196F3D;"> </span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span>s<span style="color:#196F3D;"> </span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">6</span><span style="color:#196F3D;">7</span><span style="color:#196F3D;">3</span><span style="color:#196F3D;">2</span><span style="color:#196F3D;">6</span><span style="color:#196F3D;">3</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">9</span><span style="color:#196F3D;">2</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">7</span><span style="color:#196F3D;">6</span><span style="color:#196F3D;">8</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">8</span><span style="color:#196F3D;">8</span><span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">10</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **gamma**:</code></td></tr>
    <tr style="1px solid black;"><td><code>10</code></td><td><code>11</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  <span style="color:#BA4A00;">l</span>e<span style="color:#BA4A00;">g</span><span style="color:#BA4A00;">a</span>c<span style="color:#BA4A00;">y</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">t</span>i<span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">z</span><span style="color:#BA4A00;">a</span>t<span style="color:#BA4A00;">i</span>o<span style="color:#BA4A00;">n</span> att<span style="color:#BA4A00;">r</span>i<span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">t</span>e.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">C</span><span style="color:#196F3D;">o</span>e<span style="color:#196F3D;">f</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">i</span>ci<span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span>t<span style="color:#196F3D;"> </span>o<span style="color:#196F3D;">f</span> <span style="color:#196F3D;">S</span><span style="color:#196F3D;">E</span><span style="color:#196F3D;">L</span><span style="color:#196F3D;">U</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">f</span>a<span style="color:#196F3D;">u</span><span style="color:#196F3D;">l</span>t<span style="color:#196F3D;"> </span>t<span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">0</span><span style="color:#196F3D;">5</span><span style="color:#196F3D;">0</span><span style="color:#196F3D;">7</span><span style="color:#196F3D;">0</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">0</span><span style="color:#196F3D;">2</span><span style="color:#196F3D;">2</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">4</span><span style="color:#196F3D;">8</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">3</span><span style="color:#196F3D;">2</span><span style="color:#196F3D;">3</span><span style="color:#196F3D;">2</span><span style="color:#196F3D;">4</span><span style="color:#196F3D;">2</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">8</span><span style="color:#196F3D;">7</span><span style="color:#196F3D;">5</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">(</span>i<span style="color:#196F3D;">.</span>e.<span style="color:#196F3D;">,</span></code></td></tr>
    <tr style="1px solid black;"><td><code>11</code></td><td><code>12</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">*</span> <span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">g</span>am<span style="color:#BA4A00;">m</span>a<span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">:</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code> <span style="color:#196F3D;"> </span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span>a<span style="color:#196F3D;">t</span><span style="color:#196F3D;">3</span><span style="color:#196F3D;">2</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">i</span>ma<span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">0</span><span style="color:#196F3D;">5</span><span style="color:#196F3D;">0</span><span style="color:#196F3D;">7</span><span style="color:#196F3D;">0</span><span style="color:#196F3D;">0</span><span style="color:#196F3D;">9</span><span style="color:#196F3D;">8</span><span style="color:#196F3D;">7</span><span style="color:#196F3D;">3</span><span style="color:#196F3D;">5</span><span style="color:#196F3D;">5</span><span style="color:#196F3D;">4</span><span style="color:#196F3D;">8</span><span style="color:#196F3D;">0</span><span style="color:#196F3D;">4</span><span style="color:#196F3D;">9</span><span style="color:#196F3D;">3</span><span style="color:#196F3D;">4</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">9</span><span style="color:#196F3D;">3</span><span style="color:#196F3D;">3</span><span style="color:#196F3D;">4</span><span style="color:#196F3D;">9</span><span style="color:#196F3D;">8</span><span style="color:#196F3D;">5</span><span style="color:#196F3D;">2</span><span style="color:#196F3D;">9</span><span style="color:#196F3D;">4</span><span style="color:#196F3D;">6</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">D</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">0</span><span style="color:#196F3D;">5</span><span style="color:#196F3D;">0</span><span style="color:#196F3D;">7</span><span style="color:#196F3D;">0</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">0</span><span style="color:#196F3D;">2</span><span style="color:#196F3D;">2</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">4</span><span style="color:#196F3D;">8</span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">3</span><span style="color:#196F3D;">2</span><span style="color:#196F3D;">3</span><span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">12</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  Coefficient of SELU default to 1.0507. Default value is 1.0506999492645264.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    </table>

.. _l-onnx-op-selu-1:

Selu - 1
========

**Version**

* **name**: `Selu (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Selu>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: False

This version of the operator has been available
**since version 1**.

**Summary**

Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.

**Attributes**

* **alpha**:
  Coefficient of SELU default to 1.6732. Default value is ``1.673200011253357``.
* **consumed_inputs**:
  legacy optimization attribute.
* **gamma**:
  Coefficient of SELU default to 1.0507. Default value is ``1.0506999492645264``.

**Inputs**

* **X** (heterogeneous) - **T**:
  Input tensor

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
