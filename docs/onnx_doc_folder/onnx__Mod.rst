
.. _l-onnx-doc-Mod:

===
Mod
===

.. contents::
    :local:


.. _l-onnx-op-mod-13:

Mod - 13
========

**Version**

* **name**: `Mod (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mod>`_
* **domain**: **main**
* **since_version**: **13**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 13**.

**Summary**

Performs element-wise binary modulus (with Numpy-style broadcasting support).
  The sign of the remainder is the same as that of the Divisor.

  Mod operator can also behave like C fmod() or numpy.fmod. In this case, the sign of the remainder however, will be the same as the Dividend
  (in contrast to integer mod). To force a behavior like numpy.fmod() an 'fmod' Attribute is provided.
  This attribute is set to 0 by default causing the behavior to be like integer mod.
  Setting this attribute to 1 causes the remainder to be calculated similar to that of numpy.fmod().

  If the input type is floating point, then `fmod` attribute must be set to 1.

  In case of dividend being zero, the results will be platform dependent.

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check `Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>`_.

**Attributes**

* **fmod**:
  Whether the operator should behave like fmod (default=0 meaning it
  will do integer mods); Set this to 1 to force fmod treatment Default value is ``0``.

**Inputs**

* **A** (heterogeneous) - **T**:
  Dividend tensor
* **B** (heterogeneous) - **T**:
  Divisor tensor

**Outputs**

* **C** (heterogeneous) - **T**:
  Remainder tensor

**Type Constraints**

* **T** in (
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
  Constrain input and output types to high-precision numeric tensors.

**Examples**

**mod_mixed_sign_float64**

::

    node = onnx.helper.make_node(
        'Mod',
        inputs=['x', 'y'],
        outputs=['z'],
        fmod=1
    )

    x = np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0]).astype(np.float64)
    y = np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0]).astype(np.float64)
    z = np.fmod(x, y)  # expected output [-0.1,  0.4,  5. ,  0.1, -0.4,  3.]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_mod_mixed_sign_float64')

**mod_mixed_sign_float32**

::

    node = onnx.helper.make_node(
        'Mod',
        inputs=['x', 'y'],
        outputs=['z'],
        fmod=1
    )

    x = np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0]).astype(np.float32)
    y = np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0]).astype(np.float32)
    z = np.fmod(x, y)  # expected output [-0.10000038, 0.39999962, 5. , 0.10000038, -0.39999962, 3.]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_mod_mixed_sign_float32')

**mod_mixed_sign_float16**

::

    node = onnx.helper.make_node(
        'Mod',
        inputs=['x', 'y'],
        outputs=['z'],
        fmod=1
    )

    x = np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0]).astype(np.float16)
    y = np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0]).astype(np.float16)
    z = np.fmod(x, y)  # expected output [-0.10156, 0.3984 , 5. , 0.10156, -0.3984 ,  3.]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_mod_mixed_sign_float16')

**mod_mixed_sign_int64**

::

    node = onnx.helper.make_node(
        'Mod',
        inputs=['x', 'y'],
        outputs=['z'],
    )

    x = np.array([-4, 7, 5, 4, -7, 8]).astype(np.int64)
    y = np.array([2, -3, 8, -2, 3, 5]).astype(np.int64)
    z = np.mod(x, y)  # expected output [ 0, -2,  5,  0,  2,  3]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_mod_mixed_sign_int64')

**mod_mixed_sign_int32**

::

    node = onnx.helper.make_node(
        'Mod',
        inputs=['x', 'y'],
        outputs=['z'],
    )

    x = np.array([-4, 7, 5, 4, -7, 8]).astype(np.int32)
    y = np.array([2, -3, 8, -2, 3, 5]).astype(np.int32)
    z = np.mod(x, y)  # expected output [ 0, -2,  5,  0,  2,  3]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_mod_mixed_sign_int32')

**mod_mixed_sign_int16**

::

    node = onnx.helper.make_node(
        'Mod',
        inputs=['x', 'y'],
        outputs=['z'],
    )

    x = np.array([-4, 7, 5, 4, -7, 8]).astype(np.int16)
    y = np.array([2, -3, 8, -2, 3, 5]).astype(np.int16)
    z = np.mod(x, y)  # expected output [ 0, -2,  5,  0,  2,  3]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_mod_mixed_sign_int16')

**mod_mixed_sign_int8**

::

    node = onnx.helper.make_node(
        'Mod',
        inputs=['x', 'y'],
        outputs=['z'],
    )

    x = np.array([-4, 7, 5, 4, -7, 8]).astype(np.int8)
    y = np.array([2, -3, 8, -2, 3, 5]).astype(np.int8)
    z = np.mod(x, y)  # expected output [ 0, -2,  5,  0,  2,  3]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_mod_mixed_sign_int8')

**mod_uint8**

::

    node = onnx.helper.make_node(
        'Mod',
        inputs=['x', 'y'],
        outputs=['z'],
    )

    x = np.array([4, 7, 5]).astype(np.uint8)
    y = np.array([2, 3, 8]).astype(np.uint8)
    z = np.mod(x, y)  # expected output [0, 1, 5]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_mod_uint8')

**mod_uint16**

::

    node = onnx.helper.make_node(
        'Mod',
        inputs=['x', 'y'],
        outputs=['z'],
    )

    x = np.array([4, 7, 5]).astype(np.uint16)
    y = np.array([2, 3, 8]).astype(np.uint16)
    z = np.mod(x, y)  # expected output [0, 1, 5]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_mod_uint16')

**mod_uint32**

::

    node = onnx.helper.make_node(
        'Mod',
        inputs=['x', 'y'],
        outputs=['z'],
    )

    x = np.array([4, 7, 5]).astype(np.uint32)
    y = np.array([2, 3, 8]).astype(np.uint32)
    z = np.mod(x, y)  # expected output [0, 1, 5]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_mod_uint32')

**mod_uint64**

::

    node = onnx.helper.make_node(
        'Mod',
        inputs=['x', 'y'],
        outputs=['z'],
    )

    x = np.array([4, 7, 5]).astype(np.uint64)
    y = np.array([2, 3, 8]).astype(np.uint64)
    z = np.mod(x, y)  # expected output [0, 1, 5]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_mod_uint64')

**mod_int64_fmod**

::

    node = onnx.helper.make_node(
        'Mod',
        inputs=['x', 'y'],
        outputs=['z'],
        fmod=1
    )

    x = np.array([-4, 7, 5, 4, -7, 8]).astype(np.int64)
    y = np.array([2, -3, 8, -2, 3, 5]).astype(np.int64)
    z = np.fmod(x, y)  # expected output [ 0,  1,  5,  0, -1,  3]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_mod_int64_fmod')

**mod_broadcast**

::

    node = onnx.helper.make_node(
        'Mod',
        inputs=['x', 'y'],
        outputs=['z'],
    )

    x = np.arange(0, 30).reshape([3, 2, 5]).astype(np.int32)
    y = np.array([7]).astype(np.int32)
    z = np.mod(x, y)
    #   array([[[0, 1, 2, 3, 4],
    #     [5, 6, 0, 1, 2]],

    #    [[3, 4, 5, 6, 0],
    #     [1, 2, 3, 4, 5]],

    #    [[6, 0, 1, 2, 3],
    #     [4, 5, 6, 0, 1]]], dtype=int32)
    expect(node, inputs=[x, y], outputs=[z],
           name='test_mod_broadcast')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Performs element-wise binary modulus (with Numpy-style broadcasting support).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Performs element-wise binary modulus (with Numpy-style broadcasting support).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The sign of the remainder is the same as that of the Divisor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The sign of the remainder is the same as that of the Divisor.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Mod operator can also behave like C fmod() or numpy.fmod. In this case, the sign of the remainder however, will be the same as the Dividend</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Mod operator can also behave like C fmod() or numpy.fmod. In this case, the sign of the remainder however, will be the same as the Dividend</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (in contrast to integer mod). To force a behavior like numpy.fmod() an 'fmod' Attribute is provided.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (in contrast to integer mod). To force a behavior like numpy.fmod() an 'fmod' Attribute is provided.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  This attribute is set to 0 by default causing the behavior to be like integer mod.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  This attribute is set to 0 by default causing the behavior to be like integer mod.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Setting this attribute to 1 causes the remainder to be calculated similar to that of numpy.fmod().</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Setting this attribute to 1 causes the remainder to be calculated similar to that of numpy.fmod().</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  If the input type is floating point, then fmod attribute must be set to 1.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  If the input type is floating point, then fmod attribute must be set to 1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  In case of dividend being zero, the results will be platform dependent.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  In case of dividend being zero, the results will be platform dependent.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>_.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>_.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **fmod**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **fmod**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether the operator should behave like fmod (default=0 meaning it</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether the operator should behave like fmod (default=0 meaning it</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  will do integer mods); Set this to 1 to force fmod treatment Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  will do integer mods); Set this to 1 to force fmod treatment Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **A** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **A** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Dividend tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Dividend tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Divisor tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Divisor tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **C** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **C** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Remainder tensor</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Remainder tensor</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">35</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(bfloat16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to high-precision numeric tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to high-precision numeric tensors.</code></td></tr>
    </table>

.. _l-onnx-op-mod-10:

Mod - 10
========

**Version**

* **name**: `Mod (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mod>`_
* **domain**: **main**
* **since_version**: **10**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 10**.

**Summary**

Performs element-wise binary modulus (with Numpy-style broadcasting support).
  The sign of the remainder is the same as that of the Divisor.

  Mod operator can also behave like C fmod() or numpy.fmod. In this case, the sign of the remainder however, will be the same as the Dividend
  (in contrast to integer mod). To force a behavior like numpy.fmod() an 'fmod' Attribute is provided.
  This attribute is set to 0 by default causing the behavior to be like integer mod.
  Setting this attribute to 1 causes the remainder to be calculated similar to that of numpy.fmod().

  If the input type is floating point, then `fmod` attribute must be set to 1.

  In case of dividend being zero, the results will be platform dependent.

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check `Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>`_.

**Attributes**

* **fmod**:
  Whether the operator should behave like fmod (default=0 meaning it
  will do integer mods); Set this to 1 to force fmod treatment Default value is ``0``.

**Inputs**

* **A** (heterogeneous) - **T**:
  Dividend tensor
* **B** (heterogeneous) - **T**:
  Divisor tensor

**Outputs**

* **C** (heterogeneous) - **T**:
  Remainder tensor

**Type Constraints**

* **T** in (
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
  Constrain input and output types to high-precision numeric tensors.
