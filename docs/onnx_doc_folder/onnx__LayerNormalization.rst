
.. _l-onnx-doc-LayerNormalization:

==================
LayerNormalization
==================

.. contents::
    :local:


.. _l-onnx-op-layernormalization-17:

LayerNormalization - 17
=======================

**Version**

* **name**: `LayerNormalization (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#LayerNormalization>`_
* **domain**: **main**
* **since_version**: **17**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 17**.

**Summary**

This is layer normalization defined in ONNX as function.
The overall computation can be split into two stages.
The first stage is standardization, which makes the
normalized elements have zero mean and unit variances.
The computation required by standardization can be
described by the following equations.
::

    Mean = ReduceMean<axes=normalized_axes>(X)
    D = Sub(X, Mean)
    DD = Mul(Diff, Diff)
    Var = ReduceMean<axes=normalized_axes>(DD)
    VarEps = Add(Var, epsilon)
    StdDev = Sqrt(VarEps)
    InvStdDev = Reciprocal(StdDev)
    Normalized = Mul(D, InvStdDev)

where `normalized_axes` is `[axis, ..., rank of X - 1]`.
The variables `Var` and `StdDev` stand for variance and
standard deviation, respectively. The second output is
`Mean` and the last one is `InvStdDev`.
Depending on `stash_type` attribute, the actual computation
must happen in different floating-point precision.
For example, if `stash_type` is 1, this operator casts
all input variables to 32-bit float, perform the computation, and
finally cast `Normalized` back to the original type of `X`.
The second stage then scales and shifts the outcome of the
first stage using
::

    NormalizedScaled = Mul(Normalized, Scale)
    Y = Add(NormalizedScaled, B)

The second stage doesn't depends on `stash_type`.
All equations are in [this syntax](https://github.com/onnx/onnx/blob/main/docs/Syntax.md).
The same variable (i.e., input, output, and attribute) uses
the same name in the equations above and this operator's definition.
Let `d[i]` indicate the i-th dimension of `X`.
If `X`'s shape is `[d[0], ..., d[axis-1], d[axis], ..., d[rank-1]]`,
the shape of `Mean` and `InvStdDev` is `[d[0], ..., d[axis-1], 1, ..., 1]`.
`Y` and `X` have the same shape.

**Attributes**

* **axis**:
  The first normalization dimension. If rank(X) is r, axis' allowed
  range is [-r, r]. Negative value means counting dimensions from the
  back. Default value is ``-1``.
* **epsilon**:
  The epsilon value to use to avoid division by zero. Default value is ``9.999999747378752e-06``.
* **stash_type**:
  Type of Mean and InvStdDev. This also specifies stage one's
  computation precision. Default value is ``1``.

**Inputs**

Between 2 and 3 inputs.

* **X** (heterogeneous) - **T**:
  Tensor to be normalized.
* **Scale** (heterogeneous) - **T**:
  Scale tensor.
* **B** (optional, heterogeneous) - **T**:
  Bias tensor.

**Outputs**

Between 1 and 3 outputs.

* **Y** (heterogeneous) - **T**:
  Normalized tensor.
* **Mean** (optional, heterogeneous) - **U**:
  Saved mean used during training to speed up gradient computation
* **InvStdDev** (optional, heterogeneous) - **U**:
  Saved inverse standard deviation used during training to speed up
  gradient computation.

**Type Constraints**

* **T** in (
  tensor(bfloat16),
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input types and output Y type to float tensors.
* **U** in (
  tensor(bfloat16),
  tensor(float)
  ):
  Type of Mean and InvStdDev tensors.

**Examples**

**default_axis**

::

    X = np.random.randn(2, 3, 4, 5).astype(np.float32)

    # Default axis in LayerNormalization is -1.
    normalized_shape = calculate_normalized_shape(X.shape, -1)
    W = np.random.randn(*normalized_shape).astype(np.float32)
    B = np.random.randn(*normalized_shape).astype(np.float32)
    # Axis is default to -1 in the reference implementation.
    Y, mean, inv_std_dev = _layer_normalization(X, W, B)

    # Not specifying axis attribute means -1.
    node = onnx.helper.make_node(
        'LayerNormalization',
        inputs=['X', 'W', 'B'],
        outputs=['Y', 'Mean', 'InvStdDev']
    )

    expect(node, inputs=[X, W, B], outputs=[Y, mean, inv_std_dev],
            name='test_layer_normalization_default_axis')
