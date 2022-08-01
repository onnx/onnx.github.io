
.. _l-onnx-doc-Det:

===
Det
===

.. contents::
    :local:


.. _l-onnx-op-det-11:

Det - 11
========

**Version**

* **name**: `Det (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Det>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

Det calculates determinant of a square matrix or batches of square matrices.
Det takes one input tensor of shape `[*, M, M]`, where `*` is zero or more batch dimensions,
and the inner-most 2 dimensions form square matrices.
The output is a tensor of shape `[*]`, containing the determinants of all input submatrices.
e.g., When the input is 2-D, the output is a scalar(shape is empty: `[]`).

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
  Constrain input and output types to floating-point tensors.

**Examples**

**2d**

::

    node = onnx.helper.make_node(
        'Det',
        inputs=['x'],
        outputs=['y'],
    )

    x = np.arange(4).reshape(2, 2).astype(np.float32)
    y = np.linalg.det(x)  # expect -2
    expect(node, inputs=[x], outputs=[y],
           name='test_det_2d')

**nd**

::

    node = onnx.helper.make_node(
        'Det',
        inputs=['x'],
        outputs=['y'],
    )

    x = np.array([[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]).astype(np.float32)
    y = np.linalg.det(x)  # expect array([-2., -3., -8.])
    expect(node, inputs=[x], outputs=[y],
           name='test_det_nd')
