
.. _l-onnx-doc-Einsum:

======
Einsum
======

.. contents::
    :local:


.. _l-onnx-op-einsum-12:

Einsum - 12
===========

**Version**

* **name**: `Einsum (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Einsum>`_
* **domain**: **main**
* **since_version**: **12**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 12**.

**Summary**

An einsum of the form ```term1, term2 -> output-term``` produces an output tensor using the following equation

::

    where the reduce-sum performs a summation over all the indices occurring in the input terms (term1, term2)
    that do not occur in the output-term.

    The Einsum operator evaluates algebraic tensor operations on a sequence of tensors, using the Einstein summation
    convention. The equation string contains a comma-separated sequence of lower case letters. Each term corresponds to
    an operand tensor, and the characters within the terms correspond to operands dimensions.

    This sequence may be followed by "->" to separate the left and right hand side of the equation.
    If the equation contains "->" followed by the right-hand side, the explicit (not classical) form of the Einstein
    summation is performed, and the right-hand side indices indicate output tensor dimensions. In other cases,
    output indices are (implicitly) set to the alphabetically sorted sequence of indices appearing exactly once in the
    equation.

    When a dimension character is repeated in the left-hand side, it represents summation along the dimension.

    The equation may contain ellipsis ("...") to enable broadcasting. Ellipsis must indicate a fixed number of dimensions.
    Specifically, every occurrence of ellipsis in the equation must represent the same number of dimensions.
    The right-hand side may contain exactly one ellipsis. In implicit mode, the ellipsis dimensions are set to the
    beginning of the output. The equation string may contain space (U+0020) character.

**Attributes**

* **equation** (required):
  Einsum expression string.

**Inputs**

Between 1 and 2147483647 inputs.

* **Inputs** (variadic, heterogeneous) - **T**:
  Operands

**Outputs**

* **Output** (heterogeneous) - **T**:
  Output tensor

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
  Constrain input and output types to all numerical tensor types.

**Examples**

**einsum_transpose**

::

    Eqn = 'ij->ji'
    node = onnx.helper.make_node(
        'Einsum',
        inputs=['x'],
        outputs=['y'],
        equation=Eqn
    )

    X = np.random.randn(3, 4)
    Y = einsum_reference_implementation(Eqn, (X,))

    expect(node, inputs=[X], outputs=[Y], name='test_einsum_transpose')

**einsum_sum**

::

    Eqn = 'ij->i'
    node = onnx.helper.make_node(
        'Einsum',
        inputs=['x'],
        outputs=['y'],
        equation=Eqn
    )

    X = np.random.randn(3, 4)
    Z = einsum_reference_implementation(Eqn, (X,))

    expect(node, inputs=[X], outputs=[Z], name='test_einsum_sum')

**einsum_batch_diagonal**

::

    Eqn = '...ii ->...i'
    node = onnx.helper.make_node(
        'Einsum',
        inputs=['x'],
        outputs=['y'],
        equation=Eqn
    )

    X = np.random.randn(3, 5, 5)
    Z = einsum_reference_implementation(Eqn, (X,))

    expect(node, inputs=[X], outputs=[Z], name='test_einsum_batch_diagonal')

**einsum_inner_prod**

::

    Eqn = 'i,i'
    node = onnx.helper.make_node(
        'Einsum',
        inputs=['x', 'y'],
        outputs=['z'],
        equation=Eqn
    )

    X = np.random.randn(5)
    Y = np.random.randn(5)
    Z = einsum_reference_implementation(Eqn, (X, Y))

    expect(node, inputs=[X, Y], outputs=[Z], name='test_einsum_inner_prod')

**einsum_batch_matmul**

::

    Eqn = 'bij, bjk -> bik'
    node = onnx.helper.make_node(
        'Einsum',
        inputs=['x', 'y'],
        outputs=['z'],
        equation=Eqn
    )

    X = np.random.randn(5, 2, 3)
    Y = np.random.randn(5, 3, 4)
    Z = einsum_reference_implementation(Eqn, (X, Y))

    expect(node, inputs=[X, Y], outputs=[Z], name='test_einsum_batch_matmul')
