
.. _l-onnx-doc-GlobalMaxPool:

=============
GlobalMaxPool
=============

.. contents::
    :local:


.. _l-onnx-op-globalmaxpool-1:

GlobalMaxPool - 1
=================

**Version**

* **name**: `GlobalMaxPool (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalMaxPool>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1**.

**Summary**

GlobalMaxPool consumes an input tensor X and applies max pooling across
the values in the same channel. This is equivalent to MaxPool with kernel size
equal to the spatial dimension of input tensor.

**Inputs**

* **X** (heterogeneous) - **T**:
  Input data tensor from the previous operator; dimensions for image
  case are (N x C x H x W), where N is the batch size, C is the number
  of channels, and H and W are the height and the width of the data.
  For non image case, the dimensions are in the form of (N x C x D1 x
  D2 ... Dn), where N is the batch size.

**Outputs**

* **Y** (heterogeneous) - **T**:
  Output data tensor from pooling across the input tensor. The output
  tensor has the same rank as the input. The first two dimensions of
  output shape are the same as the input (N x C), while the other
  dimensions are all 1.

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.

**Examples**

**globalmaxpool_precomputed**

::

    node = onnx.helper.make_node(
        'GlobalMaxPool',
        inputs=['x'],
        outputs=['y'],
    )
    x = np.array([[[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]]]).astype(np.float32)
    y = np.array([[[[9]]]]).astype(np.float32)
    expect(node, inputs=[x], outputs=[y], name='test_globalmaxpool_precomputed')
