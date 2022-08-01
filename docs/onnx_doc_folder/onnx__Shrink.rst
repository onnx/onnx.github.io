
.. _l-onnx-doc-Shrink:

======
Shrink
======

.. contents::
    :local:


.. _l-onnx-op-shrink-9:

Shrink - 9
==========

**Version**

* **name**: `Shrink (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shrink>`_
* **domain**: **main**
* **since_version**: **9**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 9**.

**Summary**

Shrink takes one input data (Tensor<numeric>) and produces one Tensor output,
having same datatype and shape with input. It has two attributes, lambd and
bias. The formula of this operator is: If x < -lambd, y = x + bias;
If x > lambd, y = x - bias; Otherwise, y = 0.

**Attributes**

* **bias**:
  The bias value added to output. Default is 0. Default value is ``0.0``.
* **lambd**:
  The lambd value for the Shrink formulation. Default is 0.5. Default value is ``0.5``.

**Inputs**

* **input** (heterogeneous) - **T**:
  The input data as Tensor.

**Outputs**

* **output** (heterogeneous) - **T**:
  The output.

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
  Constrain input to only numeric types.

**Examples**

**hard_shrink**

::

    node = onnx.helper.make_node(
        'Shrink',
        inputs=['x'],
        outputs=['y'],
        lambd=1.5,
    )
    X = np.arange(-2.0, 2.1, dtype=np.float32)
    Y = np.array([-2, 0, 0, 0, 2], dtype=np.float32)
    expect(node, inputs=[X], outputs=[Y],
           name='test_shrink_hard')

**soft_shrink**

::

    node = onnx.helper.make_node(
        'Shrink',
        inputs=['x'],
        outputs=['y'],
        lambd=1.5,
        bias=1.5,
    )
    X = np.arange(-2.0, 2.1, dtype=np.float32)
    Y = np.array([-0.5, 0, 0, 0, 0.5], dtype=np.float32)
    expect(node, inputs=[X], outputs=[Y],
           name='test_shrink_soft')
