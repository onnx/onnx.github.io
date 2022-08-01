
.. _l-onnx-doccom.microsoft-Trilu:

=====================
com.microsoft - Trilu
=====================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-trilu-1:

Trilu - 1 (com.microsoft)
=========================

**Version**

* **name**: `Trilu (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.Trilu>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

Returns the upper or lower triangular part of a 2-D matrix, or batches of 2-D matrices. If the attribute "upper" is set to true,
the upper triangular matrix is retained. Lower triangular matrix is retained otherwise. Default value for upper is true.
Trilu takes one input tensor of shape [*, N, M], where * is zero or more batch dimensions. The upper triangular part consists
of the elements on and above the given diagonal (k). The lower triangular part consists of elements on and below the diagonal.
All other elements in the matrix are set to zero.
If k = 0, the triangular part on and above/below the main diagonal is retained.
If upper is set to true, a positive k retains the upper triangular matrix excluding k diagonals above
the main diagonal. A negative k value includes as many diagonals below the main diagonal.
If upper is set to false, a positive k retains the lower triangular matrix including k diagonals above
the main diagonal. A negative k value excludes as many diagonals below the main diagonal.

**Attributes**

* **upper**:
  Boolean. Indicates whether upper or lower part of matrix is
  retained. Default is true. Default value is ``?``.

**Inputs**

Between 1 and 2 inputs.

* **X** (heterogeneous) - **T**:
  Input tensor of rank 2 or higher.
* **k** (optional, heterogeneous) - **tensor(int64)**:
  A 0-D tensor containing a single value corresponding to the number
  diagonals above or the main diagonal to exclude or include.Default
  value is 0 if it's not specified.

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output tensor of the same type and shape as the input tensor.

**Examples**

**triu**

::

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['x'],
        outputs=['y'],
    )

    x = np.random.randint(10, size=(4, 5)).astype(np.int64)
    # X:
    #  [[4, 7, 3, 7, 9],
    #   [1, 2, 8, 6, 9],
    #   [9, 4, 0, 8, 7],
    #   [4, 3, 4, 2, 4]]
    # expect result:
    #  [[4, 7, 3, 7, 9],
    #   [0, 2, 8, 6, 9],
    #   [0, 0, 0, 8, 7],
    #   [0, 0, 0, 2, 4]]
    y = triu_reference_implementation(x)
    expect(node, inputs=[x], outputs=[y], name='test_triu')

**triu_neg**

::

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['x', 'k'],
        outputs=['y'],
    )

    x = np.random.randint(10, size=(4, 5)).astype(np.int64)
    k = np.array(-1).astype(np.int64)
    # X:
    #  [[4, 7, 3, 7, 9],
    #   [1, 2, 8, 6, 9],
    #   [9, 4, 0, 8, 7],
    #   [4, 3, 4, 2, 4]]
    # expect result:
    #  [[4, 7, 3, 7, 9],
    #   [1, 2, 8, 6, 9],
    #   [0, 4, 0, 8, 7],
    #   [0, 0, 4, 2, 4]]
    y = triu_reference_implementation(x, int(k))
    expect(node, inputs=[x, k], outputs=[y], name='test_triu_neg')

**triu_out_neg_out**

::

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['x', 'k'],
        outputs=['y'],
    )

    x = np.random.randint(10, size=(4, 5)).astype(np.int64)
    k = np.array(-7).astype(np.int64)
    # X:
    #  [[4, 7, 3, 7, 9],
    #   [1, 2, 8, 6, 9],
    #   [9, 4, 0, 8, 7],
    #   [4, 3, 4, 2, 4]]
    # expect result:
    #  [[4, 7, 3, 7, 9],
    #   [1, 2, 8, 6, 9],
    #   [9, 4, 0, 8, 7],
    #   [4, 3, 4, 2, 4]]
    y = triu_reference_implementation(x, int(k))
    expect(node, inputs=[x, k], outputs=[y], name='test_triu_out_neg_out')

**triu_pos**

::

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['x', 'k'],
        outputs=['y'],
    )

    x = np.random.randint(10, size=(4, 5)).astype(np.int64)
    k = np.array(2).astype(np.int64)
    # X:
    #  [[4, 7, 3, 7, 9],
    #   [1, 2, 8, 6, 9],
    #   [9, 4, 0, 8, 7],
    #   [4, 3, 4, 2, 4]]
    # expect result:
    #  [[0, 0, 3, 7, 9],
    #   [0, 0, 0, 6, 9],
    #   [0, 0, 0, 0, 7],
    #   [0, 0, 0, 0, 0]]
    y = triu_reference_implementation(x, int(k))
    expect(node, inputs=[x, k], outputs=[y], name='test_triu_pos')

**triu_out_pos**

::

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['x', 'k'],
        outputs=['y'],
    )

    x = np.random.randint(10, size=(4, 5)).astype(np.int64)
    k = np.array(6).astype(np.int64)
    # X:
    #  [[4, 7, 3, 7, 9],
    #   [1, 2, 8, 6, 9],
    #   [9, 4, 0, 8, 7],
    #   [4, 3, 4, 2, 4]]
    # expect result:
    #  [[0, 0, 0, 0, 0],
    #   [0, 0, 0, 0, 0],
    #   [0, 0, 0, 0, 0],
    #   [0, 0, 0, 0, 0]]
    y = triu_reference_implementation(x, int(k))
    expect(node, inputs=[x, k], outputs=[y], name='test_triu_out_pos')

**triu_square**

::

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['x'],
        outputs=['y'],
    )

    x = np.random.randint(10, size=(2, 3, 3)).astype(np.int64)
    y = triu_reference_implementation(x)
    # X:
    # [[[4, 6, 9],
    #   [7, 5, 4],
    #   [8, 1, 2]],
    #
    #  [[1, 4, 9],
    #   [9, 6, 3],
    #   [8, 9, 8]]]
    # expect result:
    # [[[4, 6, 9],
    #   [0, 5, 4],
    #   [0, 0, 2]],
    #
    #  [[1, 4, 9],
    #   [0, 6, 3],
    #   [0, 0, 8]]]
    expect(node, inputs=[x], outputs=[y], name='test_triu_square')

**triu_square_neg**

::

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['x', 'k'],
        outputs=['y'],
    )

    x = np.random.randint(10, size=(2, 3, 3)).astype(np.int64)
    k = np.array(-1).astype(np.int64)
    # X:
    # [[[4, 6, 9],
    #   [7, 5, 4],
    #   [8, 1, 2]],
    #
    #  [[1, 4, 9],
    #   [9, 6, 3],
    #   [8, 9, 8]]]
    # expect result:
    # [[[4, 6, 9],
    #   [7, 5, 4],
    #   [0, 1, 2]],
    #
    #  [[1, 4, 9],
    #   [9, 6, 3],
    #   [0, 9, 8]]]
    y = triu_reference_implementation(x, int(k))
    expect(node, inputs=[x, k], outputs=[y], name='test_triu_square_neg')

**triu_one_row**

::

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['x', 'k'],
        outputs=['y'],
    )

    x = np.random.randint(10, size=(3, 1, 5)).astype(np.int64)
    k = np.array(1).astype(np.int64)
    # X:
    # [[[1, 4, 9, 7, 1]],
    #
    #  [[9, 2, 8, 8, 4]],
    #
    #  [[3, 9, 7, 4, 2]]]
    # expect result:
    # [[[0, 4, 9, 7, 1]],
    #
    #  [[0, 2, 8, 8, 4]],
    #
    #  [[0, 9, 7, 4, 2]]]
    y = triu_reference_implementation(x, int(k))
    expect(node, inputs=[x, k], outputs=[y], name='test_triu_one_row')

**triu_zero**

::

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['x', 'k'],
        outputs=['y'],
    )

    x = np.random.randint(10, size=(0, 5)).astype(np.int64)
    k = np.array(6).astype(np.int64)
    # X:
    # []
    # expect result:
    # []
    y = triu_reference_implementation(x, int(k))
    expect(node, inputs=[x, k], outputs=[y], name='test_triu_zero')

**tril**

::

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['x'],
        outputs=['y'],
        upper=0,
    )

    x = np.random.randint(10, size=(4, 5)).astype(np.int64)
    # X:
    #  [[4, 7, 3, 7, 9],
    #   [1, 2, 8, 6, 9],
    #   [9, 4, 1, 8, 7],
    #   [4, 3, 4, 2, 4]]
    # expect result:
    #  [[4, 0, 0, 0, 0],
    #   [1, 2, 0, 0, 0],
    #   [9, 4, 1, 0, 0],
    #   [4, 3, 4, 2, 0]]
    y = tril_reference_implementation(x)
    expect(node, inputs=[x], outputs=[y], name='test_tril')

**tril_neg**

::

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['x', 'k'],
        outputs=['y'],
        upper=0,
    )

    x = np.random.randint(10, size=(4, 5)).astype(np.int64)
    k = np.array(-1).astype(np.int64)
    # X:
    #  [[4, 7, 3, 7, 9],
    #   [1, 2, 8, 6, 9],
    #   [9, 4, 1, 8, 7],
    #   [4, 3, 4, 2, 4]]
    # expect result:
    #  [[0, 0, 0, 0, 0],
    #   [1, 0, 0, 0, 0],
    #   [9, 4, 0, 0, 0],
    #   [4, 3, 4, 0, 0]]
    y = tril_reference_implementation(x, int(k))
    expect(node, inputs=[x, k], outputs=[y], name='test_tril_neg')

**tril_out_neg**

::

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['x', 'k'],
        outputs=['y'],
        upper=0,
    )

    x = np.random.randint(10, size=(4, 5)).astype(np.int64)
    k = np.array(-7).astype(np.int64)
    # X:
    #  [[4, 7, 3, 7, 9],
    #   [1, 2, 8, 6, 9],
    #   [9, 4, 1, 8, 7],
    #   [4, 3, 4, 2, 4]]
    # expect result:
    #  [[0, 0, 0, 0, 0],
    #   [0, 0, 0, 0, 0],
    #   [0, 0, 0, 0, 0],
    #   [0, 0, 0, 0, 0]]
    y = tril_reference_implementation(x, int(k))
    expect(node, inputs=[x, k], outputs=[y], name='test_tril_out_neg')

**tril_pos**

::

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['x', 'k'],
        outputs=['y'],
        upper=0,
    )

    x = np.random.randint(10, size=(4, 5)).astype(np.int64)
    k = np.array(2).astype(np.int64)
    # X:
    #  [[4, 7, 3, 7, 9],
    #   [1, 2, 8, 6, 9],
    #   [9, 4, 1, 8, 7],
    #   [4, 3, 4, 2, 4]]
    # expect result:
    #  [[4, 7, 3, 0, 0],
    #   [1, 2, 8, 6, 0],
    #   [9, 4, 1, 8, 7],
    #   [4, 3, 4, 2, 4]]
    y = tril_reference_implementation(x, int(k))
    expect(node, inputs=[x, k], outputs=[y], name='test_tril_pos')

**tril_out_pos**

::

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['x', 'k'],
        outputs=['y'],
        upper=0,
    )
    x = np.random.randint(10, size=(4, 5)).astype(np.int64)
    k = np.array(6).astype(np.int64)
    # X:
    #  [[4, 7, 3, 7, 9],
    #   [1, 2, 8, 6, 9],
    #   [9, 4, 1, 8, 7],
    #   [4, 3, 4, 2, 4]]
    # expect result:
    #  [[4, 7, 3, 7, 9],
    #   [1, 2, 8, 6, 9],
    #   [9, 4, 1, 8, 7],
    #   [4, 3, 4, 2, 4]]
    y = tril_reference_implementation(x, int(k))
    expect(node, inputs=[x, k], outputs=[y], name='test_tril_out_pos')

**tril_square**

::

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['x'],
        outputs=['y'],
        upper=0,
    )

    x = np.random.randint(10, size=(2, 3, 3)).astype(np.int64)
    # X:
    # [[[0, 4, 3],
    #   [2, 0, 9],
    #   [8, 2, 5]],
    #
    #  [[2, 7, 2],
    #   [2, 6, 0],
    #   [2, 6, 5]]]
    # expect result:
    # [[[0, 0, 0],
    #   [2, 0, 0],
    #   [8, 2, 5]],
    #
    #  [[2, 0, 0],
    #   [2, 6, 0],
    #   [2, 6, 5]]]
    y = tril_reference_implementation(x)
    expect(node, inputs=[x], outputs=[y], name='test_tril_square')

**tril_square_neg**

::

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['x', 'k'],
        outputs=['y'],
        upper=0,
    )

    x = np.random.randint(10, size=(2, 3, 3)).astype(np.int64)
    k = np.array(-1).astype(np.int64)
    # X:
    # [[[0, 4, 3],
    #   [2, 0, 9],
    #   [8, 2, 5]],
    #
    #  [[2, 7, 2],
    #   [2, 6, 0],
    #   [2, 6, 5]]]
    # expect result:
    # [[[0, 0, 0],
    #   [2, 0, 0],
    #   [8, 2, 0]],
    #
    #  [[0, 0, 0],
    #   [2, 0, 0],
    #   [2, 6, 0]]]
    y = tril_reference_implementation(x, int(k))
    expect(node, inputs=[x, k], outputs=[y], name='test_tril_square_neg')

**tril_one_row**

::

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['x'],
        outputs=['y'],
        upper=0,
    )

    x = np.random.randint(10, size=(3, 1, 5)).astype(np.int64)
    # X:
    # [[[6, 2, 4, 1, 6]],
    #
    #  [[8, 3, 8, 7, 0]],
    #
    #  [[2, 2, 9, 5, 9]]]
    # expect result:
    # [[[6, 0, 0, 0, 0]],
    #
    #  [[8, 0, 0, 0, 0]],
    #
    #  [[2, 0, 0, 0, 0]]]
    y = tril_reference_implementation(x)
    expect(node, inputs=[x], outputs=[y], name='test_tril_one_row_neg')

**tril_zero**

::

    node = onnx.helper.make_node(
        'Trilu',
        inputs=['x', 'k'],
        outputs=['y'],
        upper=0,
    )

    x = np.random.randint(10, size=(3, 0, 5)).astype(np.int64)
    k = np.array(6).astype(np.int64)
    # X:
    # []
    # expect result:
    # []
    y = tril_reference_implementation(x, int(k))
    expect(node, inputs=[x, k], outputs=[y], name='test_tril_zero')
