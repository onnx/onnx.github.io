
.. _l-onnx-doc-Gemm:

====
Gemm
====

.. contents::
    :local:


.. _l-onnx-op-gemm-13:

Gemm - 13
=========

**Version**

* **name**: `Gemm (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm>`_
* **domain**: **main**
* **since_version**: **13**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 13**.

**Summary**

General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check `Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>`_.
This operator has **optional** inputs/outputs. See `ONNX <https://github.com/onnx/onnx/blob/master/docs/IR.md>`_ for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

**Attributes**

* **alpha**:
  Scalar multiplier for the product of input tensors A * B. Default value is ``1.0``.
* **beta**:
  Scalar multiplier for input tensor C. Default value is ``1.0``.
* **transA**:
  Whether A should be transposed Default value is ``0``.
* **transB**:
  Whether B should be transposed Default value is ``0``.

**Inputs**

Between 2 and 3 inputs.

* **A** (heterogeneous) - **T**:
  Input tensor A. The shape of A should be (M, K) if transA is 0, or
  (K, M) if transA is non-zero.
* **B** (heterogeneous) - **T**:
  Input tensor B. The shape of B should be (K, N) if transB is 0, or
  (N, K) if transB is non-zero.
* **C** (optional, heterogeneous) - **T**:
  Optional input tensor C. If not specified, the computation is done
  as if C is a scalar 0. The shape of C should be unidirectional
  broadcastable to (M, N).

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output tensor of shape (M, N).

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

**default_zero_bias**

::

    node = onnx.helper.make_node(
        'Gemm',
        inputs=['a', 'b', 'c'],
        outputs=['y']
    )
    a = np.random.ranf([3, 5]).astype(np.float32)
    b = np.random.ranf([5, 4]).astype(np.float32)
    c = np.zeros([1, 4]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c)
    expect(node, inputs=[a, b, c], outputs=[y],
           name='test_gemm_default_zero_bias')

**default_no_bias**

::

    node = onnx.helper.make_node(
        'Gemm',
        inputs=['a', 'b'],
        outputs=['y']
    )
    a = np.random.ranf([2, 10]).astype(np.float32)
    b = np.random.ranf([10, 3]).astype(np.float32)
    y = gemm_reference_implementation(a, b)
    expect(node, inputs=[a, b], outputs=[y],
           name='test_gemm_default_no_bias')

**default_scalar_bias**

::

    node = onnx.helper.make_node(
        'Gemm',
        inputs=['a', 'b', 'c'],
        outputs=['y']
    )
    a = np.random.ranf([2, 3]).astype(np.float32)
    b = np.random.ranf([3, 4]).astype(np.float32)
    c = np.array(3.14).astype(np.float32)
    y = gemm_reference_implementation(a, b, c)
    expect(node, inputs=[a, b, c], outputs=[y],
           name='test_gemm_default_scalar_bias')

**default_single_elem_vector_bias**

::

    node = onnx.helper.make_node(
        'Gemm',
        inputs=['a', 'b', 'c'],
        outputs=['y']
    )
    a = np.random.ranf([3, 7]).astype(np.float32)
    b = np.random.ranf([7, 3]).astype(np.float32)
    c = np.random.ranf([1]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c)
    expect(node, inputs=[a, b, c], outputs=[y],
           name='test_gemm_default_single_elem_vector_bias')

**default_vector_bias**

::

    node = onnx.helper.make_node(
        'Gemm',
        inputs=['a', 'b', 'c'],
        outputs=['y']
    )
    a = np.random.ranf([2, 7]).astype(np.float32)
    b = np.random.ranf([7, 4]).astype(np.float32)
    c = np.random.ranf([1, 4]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c)
    expect(node, inputs=[a, b, c], outputs=[y],
           name='test_gemm_default_vector_bias')

**default_matrix_bias**

::

    node = onnx.helper.make_node(
        'Gemm',
        inputs=['a', 'b', 'c'],
        outputs=['y']
    )
    a = np.random.ranf([3, 6]).astype(np.float32)
    b = np.random.ranf([6, 4]).astype(np.float32)
    c = np.random.ranf([3, 4]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c)
    expect(node, inputs=[a, b, c], outputs=[y],
           name='test_gemm_default_matrix_bias')

**transposeA**

::

    node = onnx.helper.make_node(
        'Gemm',
        inputs=['a', 'b', 'c'],
        outputs=['y'],
        transA=1
    )
    a = np.random.ranf([6, 3]).astype(np.float32)
    b = np.random.ranf([6, 4]).astype(np.float32)
    c = np.zeros([1, 4]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c, transA=1)
    expect(node, inputs=[a, b, c], outputs=[y],
           name='test_gemm_transposeA')

**transposeB**

::

    node = onnx.helper.make_node(
        'Gemm',
        inputs=['a', 'b', 'c'],
        outputs=['y'],
        transB=1
    )
    a = np.random.ranf([3, 6]).astype(np.float32)
    b = np.random.ranf([4, 6]).astype(np.float32)
    c = np.zeros([1, 4]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c, transB=1)
    expect(node, inputs=[a, b, c], outputs=[y],
           name='test_gemm_transposeB')

**alpha**

::

    node = onnx.helper.make_node(
        'Gemm',
        inputs=['a', 'b', 'c'],
        outputs=['y'],
        alpha=0.5
    )
    a = np.random.ranf([3, 5]).astype(np.float32)
    b = np.random.ranf([5, 4]).astype(np.float32)
    c = np.zeros([1, 4]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c, alpha=0.5)
    expect(node, inputs=[a, b, c], outputs=[y],
           name='test_gemm_alpha')

**beta**

::

    node = onnx.helper.make_node(
        'Gemm',
        inputs=['a', 'b', 'c'],
        outputs=['y'],
        beta=0.5
    )
    a = np.random.ranf([2, 7]).astype(np.float32)
    b = np.random.ranf([7, 4]).astype(np.float32)
    c = np.random.ranf([1, 4]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c, beta=0.5)
    expect(node, inputs=[a, b, c], outputs=[y],
           name='test_gemm_beta')

**all_attributes**

::

    node = onnx.helper.make_node(
        'Gemm',
        inputs=['a', 'b', 'c'],
        outputs=['y'],
        alpha=0.25,
        beta=0.35,
        transA=1,
        transB=1
    )
    a = np.random.ranf([4, 3]).astype(np.float32)
    b = np.random.ranf([5, 4]).astype(np.float32)
    c = np.random.ranf([1, 5]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c, transA=1, transB=1, alpha=0.25, beta=0.35)
    expect(node, inputs=[a, b, c], outputs=[y],
           name='test_gemm_all_attributes')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">General Matrix multiplication:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">General Matrix multiplication:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">A' = transpose(A) if transA else A</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">A' = transpose(A) if transA else A</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">B' = transpose(B) if transB else B</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">B' = transpose(B) if transB else B</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">and output tensor Y has shape (M, N). A will be transposed before doing the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">and output tensor Y has shape (M, N). A will be transposed before doing the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">computation if attribute transA is non-zero, same for B and transB.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">computation if attribute transA is non-zero, same for B and transB.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>_.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>_.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator has **optional** inputs/outputs. See ONNX <https://github.com/onnx/onnx/blob/master/docs/IR.md>_ for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator has **optional** inputs/outputs. See ONNX <https://github.com/onnx/onnx/blob/master/docs/IR.md>_ for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **alpha**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **alpha**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Scalar multiplier for the product of input tensors A * B. Default value is 1.0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Scalar multiplier for the product of input tensors A * B. Default value is 1.0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **beta**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **beta**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Scalar multiplier for input tensor C. Default value is 1.0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Scalar multiplier for input tensor C. Default value is 1.0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **transA**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **transA**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether A should be transposed Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether A should be transposed Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **transB**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **transB**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether B should be transposed Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether B should be transposed Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 3 inputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 3 inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **A** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **A** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor A. The shape of A should be (M, K) if transA is 0, or</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor A. The shape of A should be (M, K) if transA is 0, or</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (K, M) if transA is non-zero.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (K, M) if transA is non-zero.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor B. The shape of B should be (K, N) if transB is 0, or</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor B. The shape of B should be (K, N) if transB is 0, or</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (N, K) if transB is non-zero.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (N, K) if transB is non-zero.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **C** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **C** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional input tensor C. If not specified, the computation is done</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional input tensor C. If not specified, the computation is done</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  as if C is a scalar 0. The shape of C should be unidirectional</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  as if C is a scalar 0. The shape of C should be unidirectional</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  broadcastable to (M, N).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  broadcastable to (M, N).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor of shape (M, N).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor of shape (M, N).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">48</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(bfloat16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float/int tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float/int tensors.</code></td></tr>
    </table>

.. _l-onnx-op-gemm-11:

Gemm - 11
=========

**Version**

* **name**: `Gemm (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check `Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>`_.
This operator has **optional** inputs/outputs. See `ONNX <https://github.com/onnx/onnx/blob/master/docs/IR.md>`_ for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

**Attributes**

* **alpha**:
  Scalar multiplier for the product of input tensors A * B. Default value is ``1.0``.
* **beta**:
  Scalar multiplier for input tensor C. Default value is ``1.0``.
* **transA**:
  Whether A should be transposed Default value is ``0``.
* **transB**:
  Whether B should be transposed Default value is ``0``.

**Inputs**

Between 2 and 3 inputs.

* **A** (heterogeneous) - **T**:
  Input tensor A. The shape of A should be (M, K) if transA is 0, or
  (K, M) if transA is non-zero.
* **B** (heterogeneous) - **T**:
  Input tensor B. The shape of B should be (K, N) if transB is 0, or
  (N, K) if transB is non-zero.
* **C** (optional, heterogeneous) - **T**:
  Optional input tensor C. If not specified, the computation is done
  as if C is a scalar 0. The shape of C should be unidirectional
  broadcastable to (M, N).

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output tensor of shape (M, N).

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
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">General Matrix multiplication:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">General Matrix multiplication:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">A' = transpose(A) if transA else A</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">A' = transpose(A) if transA else A</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">B' = transpose(B) if transB else B</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">B' = transpose(B) if transB else B</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">and output tensor Y has shape (M, N). A will be transposed before doing the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">and output tensor Y has shape (M, N). A will be transposed before doing the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">computation if attribute transA is non-zero, same for B and transB.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">computation if attribute transA is non-zero, same for B and transB.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>_.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>_.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">12</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">This operator has **optional** inputs/outputs. See ONNX <https://github.com/onnx/onnx/blob/master/docs/IR.md>_ for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **alpha**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **alpha**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Scalar multiplier for the product of input tensors A * B. Default value is 1.0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Scalar multiplier for the product of input tensors A * B. Default value is 1.0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **beta**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **beta**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Scalar multiplier for input tensor C. Default value is 1.0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Scalar multiplier for input tensor C. Default value is 1.0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **transA**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **transA**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether A should be transposed Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether A should be transposed Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **transB**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **transB**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether B should be transposed Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether B should be transposed Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">27</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">Between 2 and 3 inputs.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">28</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **A** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **A** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor A. The shape of A should be (M, K) if transA is 0, or</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor A. The shape of A should be (M, K) if transA is 0, or</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (K, M) if transA is non-zero.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (K, M) if transA is non-zero.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor B. The shape of B should be (K, N) if transB is 0, or</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor B. The shape of B should be (K, N) if transB is 0, or</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (N, K) if transB is non-zero.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (N, K) if transB is non-zero.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">35</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **C** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td><code>32</code></td><td><code>36</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">*</span> <span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">C</span><span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">*</span> <span style="color:#BA4A00;">(</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span>tero<span style="color:#BA4A00;">g</span>e<span style="color:#BA4A00;">n</span>eous<span style="color:#BA4A00;">)</span> <span style="color:#BA4A00;">-</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">T</span><span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">:</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">O</span><span style="color:#196F3D;">p</span>t<span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span>e<span style="color:#196F3D;">n</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">o</span>r<span style="color:#196F3D;"> </span><span style="color:#196F3D;">C</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">I</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">n</span>o<span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">p</span>e<span style="color:#196F3D;">c</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">i</span>e<span style="color:#196F3D;">d</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">c</span>o<span style="color:#196F3D;">m</span><span style="color:#196F3D;">p</span>u<span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span>s <span style="color:#196F3D;">d</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">e</span></code></td></tr>
    <tr style="1px solid black;"><td><code>33</code></td><td><code>37</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  <span style="color:#BA4A00;">I</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">t</span> <span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span>s<span style="color:#BA4A00;">o</span>r <span style="color:#BA4A00;">C</span>. The shape of C should be unidirectional</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">a</span><span style="color:#196F3D;">s</span> <span style="color:#196F3D;">i</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">C</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span>s<span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">a</span>r <span style="color:#196F3D;">0</span>. The shape of C should be unidirectional</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  broadcastable to (M, N).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  broadcastable to (M, N).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor of shape (M, N).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor of shape (M, N).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float/int tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float/int tensors.</code></td></tr>
    </table>

.. _l-onnx-op-gemm-9:

Gemm - 9
========

**Version**

* **name**: `Gemm (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm>`_
* **domain**: **main**
* **since_version**: **9**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 9**.

**Summary**

General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check `Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>`_.

**Attributes**

* **alpha**:
  Scalar multiplier for the product of input tensors A * B. Default value is ``1.0``.
* **beta**:
  Scalar multiplier for input tensor C. Default value is ``1.0``.
* **transA**:
  Whether A should be transposed Default value is ``0``.
* **transB**:
  Whether B should be transposed Default value is ``0``.

**Inputs**

* **A** (heterogeneous) - **T**:
  Input tensor A. The shape of A should be (M, K) if transA is 0, or
  (K, M) if transA is non-zero.
* **B** (heterogeneous) - **T**:
  Input tensor B. The shape of B should be (K, N) if transB is 0, or
  (N, K) if transB is non-zero.
* **C** (heterogeneous) - **T**:
  Input tensor C. The shape of C should be unidirectional
  broadcastable to (M, N).

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output tensor of shape (M, N).

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
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">General Matrix multiplication:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">General Matrix multiplication:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">A' = transpose(A) if transA else A</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">A' = transpose(A) if transA else A</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">B' = transpose(B) if transB else B</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">B' = transpose(B) if transB else B</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">and output tensor Y has shape (M, N). A will be transposed before doing the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">and output tensor Y has shape (M, N). A will be transposed before doing the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">computation if attribute transA is non-zero, same for B and transB.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">computation if attribute transA is non-zero, same for B and transB.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>_.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>_.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **alpha**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **alpha**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Scalar multiplier for the product of input tensors A * B. Default value is 1.0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Scalar multiplier for the product of input tensors A * B. Default value is 1.0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **beta**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **beta**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Scalar multiplier for input tensor C. Default value is 1.0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Scalar multiplier for input tensor C. Default value is 1.0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **transA**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **transA**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether A should be transposed Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether A should be transposed Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **transB**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **transB**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether B should be transposed Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether B should be transposed Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **A** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **A** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor A. The shape of A should be (M, K) if transA is 0, or</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor A. The shape of A should be (M, K) if transA is 0, or</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (K, M) if transA is non-zero.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (K, M) if transA is non-zero.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor B. The shape of B should be (K, N) if transB is 0, or</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor B. The shape of B should be (K, N) if transB is 0, or</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (N, K) if transB is non-zero.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (N, K) if transB is non-zero.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **C** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **C** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor C. The shape of C should be unidirectional</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor C. The shape of C should be unidirectional</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  broadcastable to (M, N).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  broadcastable to (M, N).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor of shape (M, N).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor of shape (M, N).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td><code>46</code></td><td><code>46</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  tensor(float16)</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  tensor(float16)<span style="color:#196F3D;">,</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">47</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">48</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">49</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">50</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(uint64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td><code>48</code></td><td><code>52</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Constrain input and output types to float tensors.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Constrain input and output types to float<span style="color:#196F3D;">/</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span> tensors.</code></td></tr>
    </table>

.. _l-onnx-op-gemm-7:

Gemm - 7
========

**Version**

* **name**: `Gemm (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm>`_
* **domain**: **main**
* **since_version**: **7**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 7**.

**Summary**

General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check `Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>`_.

**Attributes**

* **alpha**:
  Scalar multiplier for the product of input tensors A * B. Default value is ``1.0``.
* **beta**:
  Scalar multiplier for input tensor C. Default value is ``1.0``.
* **transA**:
  Whether A should be transposed Default value is ``0``.
* **transB**:
  Whether B should be transposed Default value is ``0``.

**Inputs**

* **A** (heterogeneous) - **T**:
  Input tensor A. The shape of A should be (M, K) if transA is 0, or
  (K, M) if transA is non-zero.
* **B** (heterogeneous) - **T**:
  Input tensor B. The shape of B should be (K, N) if transB is 0, or
  (N, K) if transB is non-zero.
* **C** (heterogeneous) - **T**:
  Input tensor C. The shape of C should be unidirectional
  broadcastable to (M, N).

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output tensor of shape (M, N).

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
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">General Matrix multiplication:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">General Matrix multiplication:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">2</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">3</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">A' = transpose(A) if transA else A</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">4</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">5</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">B' = transpose(B) if transB else B</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">6</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td><code>2</code></td><td><code>7</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">Compute Y = alpha * A * B + beta * C, where input tensor A has</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>Compute Y = alpha * A<span style="color:#196F3D;">'</span> * B<span style="color:#196F3D;">'</span> + beta * C, where input tensor A has<span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">M</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">K</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">K</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">M</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;">,</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">3</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">dimension (M X K), input tensor B has dimension (K X N), input tensor C and</code></td><td></td></tr>
    <tr style="1px solid black;"><td><code>4</code></td><td><code>8</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">t</span>put tensor <span style="color:#BA4A00;">Y</span> ha<span style="color:#BA4A00;">v</span>e <span style="color:#BA4A00;">d</span>i<span style="color:#BA4A00;">m</span>ensio<span style="color:#BA4A00;">n</span> (M <span style="color:#BA4A00;">X</span><span style="color:#BA4A00;"> </span>N)<span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span>put tensor <span style="color:#196F3D;">B</span> ha<span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">p</span>e <span style="color:#196F3D;">(</span><span style="color:#196F3D;">K</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">N</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">N</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">K</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span>i<span style="color:#196F3D;">n</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span>ens<span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">C</span><span style="color:#196F3D;"> </span>i<span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">r</span>o<span style="color:#196F3D;">a</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">e</span> <span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span>(M<span style="color:#196F3D;">,</span> N)<span style="color:#196F3D;">,</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">9</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">and output tensor Y has shape (M, N). A will be transposed before doing the</code></td></tr>
    <tr style="1px solid black;"><td><code>5</code></td><td><code>10</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">I</span>f attribute <span style="color:#BA4A00;">b</span>r<span style="color:#BA4A00;">o</span>a<span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">a</span>s<span style="color:#BA4A00;">t</span> is non-zero, <span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">t</span> <span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">s</span>or <span style="color:#BA4A00;">C</span> <span style="color:#BA4A00;">w</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">o</span>ad<span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">d</span> t<span style="color:#BA4A00;">o</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">m</span>a<span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">h</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">c</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span>f attribute <span style="color:#196F3D;">t</span>ra<span style="color:#196F3D;">n</span>s<span style="color:#196F3D;">A</span> is non-zero, <span style="color:#196F3D;">s</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span> <span style="color:#196F3D;">f</span>or <span style="color:#196F3D;">B</span> a<span style="color:#196F3D;">n</span>d t<span style="color:#196F3D;">r</span>a<span style="color:#196F3D;">n</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">B</span><span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td><code>6</code></td><td><code>11</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">t</span>he di<span style="color:#BA4A00;">m</span>ensi<span style="color:#BA4A00;">o</span>n re<span style="color:#BA4A00;">q</span>uire<span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span>t<span style="color:#BA4A00;">.</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">A</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">w</span>il<span style="color:#BA4A00;">l</span> be t<span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">a</span>ns<span style="color:#BA4A00;">p</span>o<span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">d</span> <span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">e</span>fore doing th<span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span>com<span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">u</span>tati<span style="color:#BA4A00;">o</span>n</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">T</span>h<span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span>e<span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span> <span style="color:#196F3D;">s</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">i</span>di<span style="color:#196F3D;">r</span>e<span style="color:#196F3D;">c</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span>n<span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span>s<span style="color:#196F3D;">t</span>in<span style="color:#196F3D;">g</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span> <span style="color:#196F3D;">(</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">o</span>r<span style="color:#196F3D;"> </span><span style="color:#196F3D;">C</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span>e<span style="color:#196F3D;"> </span>u<span style="color:#196F3D;">n</span>i<span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span>re<span style="color:#196F3D;">c</span>ti<span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span>l b<span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">l</span>e t<span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span>nso<span style="color:#196F3D;">r</span> <span style="color:#196F3D;">A</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">*</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">B</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;">;</span><span style="color:#196F3D;"> </span>for<span style="color:#196F3D;"> </span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span>e d<span style="color:#196F3D;">e</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">k</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">B</span><span style="color:#196F3D;">r</span>o<span style="color:#196F3D;">a</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span>ing <span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">O</span><span style="color:#196F3D;">N</span><span style="color:#196F3D;">N</span><span style="color:#196F3D;">X</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"><</span><span style="color:#196F3D;">h</span>t<span style="color:#196F3D;">t</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">:</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">t</span>h<span style="color:#196F3D;">u</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">.</span>com<span style="color:#196F3D;">/</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">s</span>t<span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">B</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span>a<span style="color:#196F3D;">d</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">s</span>tin<span style="color:#196F3D;">g</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">></span><span style="color:#196F3D;">_</span><span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">7</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">if attribute transA is non-zero, same for B and transB.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **alpha**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **alpha**:</code></td></tr>
    <tr style="1px solid black;"><td><code>12</code></td><td><code>16</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Scalar multiplier for the product of input tensors A * B<span style="color:#BA4A00;">,</span> t<span style="color:#BA4A00;">h</span>e</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Scalar multiplier for the product of input tensors A * B<span style="color:#196F3D;">.</span> <span style="color:#196F3D;">D</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">l</span>t<span style="color:#196F3D;"> </span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">u</span>e<span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">1</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">0</span><span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">13</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  default value is 1.0. Default value is 1.0.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **beta**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **beta**:</code></td></tr>
    <tr style="1px solid black;"><td><code>15</code></td><td><code>18</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Scalar multiplier for input tensor C<span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">v</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">1</span>.<span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">.</span> Default value is 1.0.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Scalar multiplier for input tensor C. Default value is 1.0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">16</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">* **broadcast**:</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">17</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  Whether C should be broadcasted Default value is 0.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **transA**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **transA**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether A should be transposed Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether A should be transposed Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **transB**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **transB**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether B should be transposed Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether B should be transposed Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **A** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **A** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td><code>26</code></td><td><code>27</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Input tensor A</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Input tensor A<span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">T</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">A</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">M</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">K</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">A</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">0</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">28</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  (K, M) if transA is non-zero.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td><code>28</code></td><td><code>30</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Input tensor B</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Input tensor B<span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">T</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">B</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">K</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">N</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">B</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">0</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">31</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  (N, K) if transB is non-zero.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **C** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **C** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td><code>30</code></td><td><code>33</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Input tensor C</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Input tensor C<span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">T</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">C</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">34</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  broadcastable to (M, N).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td><code>35</code></td><td><code>39</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Output tensor.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Output tensor<span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">M</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">N</span><span style="color:#196F3D;">)</span>.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    </table>

.. _l-onnx-op-gemm-6:

Gemm - 6
========

**Version**

* **name**: `Gemm (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm>`_
* **domain**: **main**
* **since_version**: **6**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 6**.

**Summary**

General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
Compute Y = alpha * A * B + beta * C, where input tensor A has
dimension (M X K), input tensor B has dimension (K X N), input tensor C and
output tensor Y have dimension (M X N).
If attribute broadcast is non-zero, input tensor C will be broadcasted to match
the dimension requirement. A will be transposed before doing the computation
if attribute transA is non-zero, same for B and transB.

**Attributes**

* **alpha**:
  Scalar multiplier for the product of input tensors A * B, the
  default value is 1.0. Default value is ``1.0``.
* **beta**:
  Scalar multiplier for input tensor C, the default value is 1.0. Default value is ``1.0``.
* **broadcast**:
  Whether C should be broadcasted Default value is ``0``.
* **transA**:
  Whether A should be transposed Default value is ``0``.
* **transB**:
  Whether B should be transposed Default value is ``0``.

**Inputs**

* **A** (heterogeneous) - **T**:
  Input tensor A
* **B** (heterogeneous) - **T**:
  Input tensor B
* **C** (heterogeneous) - **T**:
  Input tensor C

**Outputs**

* **Y** (heterogeneous) - **T**:
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
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">General Matrix multiplication:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">General Matrix multiplication:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Compute Y = alpha * A * B + beta * C, where input tensor A has</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Compute Y = alpha * A * B + beta * C, where input tensor A has</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">dimension (M X K), input tensor B has dimension (K X N), input tensor C and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">dimension (M X K), input tensor B has dimension (K X N), input tensor C and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">output tensor Y have dimension (M X N).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">output tensor Y have dimension (M X N).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If attribute broadcast is non-zero, input tensor C will be broadcasted to match</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If attribute broadcast is non-zero, input tensor C will be broadcasted to match</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the dimension requirement. A will be transposed before doing the computation</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the dimension requirement. A will be transposed before doing the computation</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">if attribute transA is non-zero, same for B and transB.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">if attribute transA is non-zero, same for B and transB.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **alpha**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **alpha**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Scalar multiplier for the product of input tensors A * B, the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Scalar multiplier for the product of input tensors A * B, the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  default value is 1.0. Default value is 1.0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  default value is 1.0. Default value is 1.0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **beta**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **beta**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Scalar multiplier for input tensor C, the default value is 1.0. Default value is 1.0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Scalar multiplier for input tensor C, the default value is 1.0. Default value is 1.0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **broadcast**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **broadcast**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether C should be broadcasted Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether C should be broadcasted Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **transA**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **transA**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether A should be transposed Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether A should be transposed Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **transB**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **transB**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether B should be transposed Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Whether B should be transposed Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **A** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **A** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor A</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor A</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor B</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor B</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **C** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **C** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td><code>30</code></td><td><code>30</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Input tensor C<span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Input tensor C</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Output tensor.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    </table>

.. _l-onnx-op-gemm-1:

Gemm - 1
========

**Version**

* **name**: `Gemm (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: False

This version of the operator has been available
**since version 1**.

**Summary**

General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
Compute Y = alpha * A * B + beta * C, where input tensor A has
dimension (M X K), input tensor B has dimension (K X N), input tensor C and
output tensor Y have dimension (M X N).
If attribute broadcast is non-zero, input tensor C will be broadcasted to match
the dimension requirement. A will be transposed before doing the computation
if attribute transA is non-zero, same for B and transB.

**Attributes**

* **alpha**:
  Scalar multiplier for the product of input tensors A * B, the
  default value is 1.0. Default value is ``1.0``.
* **beta**:
  Scalar multiplier for input tensor C, the default value is 1.0. Default value is ``1.0``.
* **broadcast**:
  Whether C should be broadcasted Default value is ``0``.
* **transA**:
  Whether A should be transposed Default value is ``0``.
* **transB**:
  Whether B should be transposed Default value is ``0``.

**Inputs**

* **A** (heterogeneous) - **T**:
  Input tensor A
* **B** (heterogeneous) - **T**:
  Input tensor B
* **C** (heterogeneous) - **T**:
  Input tensor C, can be inplace.

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output tensor.

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
