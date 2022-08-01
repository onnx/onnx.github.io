
.. _l-onnx-doc-BitShift:

========
BitShift
========

.. contents::
    :local:


.. _l-onnx-op-bitshift-11:

BitShift - 11
=============

**Version**

* **name**: `BitShift (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#BitShift>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

Bitwise shift operator performs element-wise operation. For each input element, if the
 attribute "direction" is "RIGHT", this operator moves its binary representation toward
 the right side so that the input value is effectively decreased. If the attribute "direction"
 is "LEFT", bits of binary representation moves toward the left side, which results the
 increase of its actual value. The input X is the tensor to be shifted and another input
 Y specifies the amounts of shifting. For example, if "direction" is "Right", X is [1, 4],
 and S is [1, 1], the corresponding output Z would be [0, 2]. If "direction" is "LEFT" with
 X=[1, 2] and S=[1, 2], the corresponding output Y would be [2, 8].

 Because this operator supports Numpy-style broadcasting, X's and Y's shapes are
 not necessarily identical.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check `Broadcasting in ONNX <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>`_.

**Attributes**

* **direction** (required):
  Direction of moving bits. It can be either "RIGHT" (for right shift)
  or "LEFT" (for left shift).

**Inputs**

* **X** (heterogeneous) - **T**:
  First operand, input to be shifted.
* **Y** (heterogeneous) - **T**:
  Second operand, amounts of shift.

**Outputs**

* **Z** (heterogeneous) - **T**:
  Output tensor

**Type Constraints**

* **T** in (
  tensor(uint16),
  tensor(uint32),
  tensor(uint64),
  tensor(uint8)
  ):
  Constrain input and output types to integer tensors.

**Examples**

**right_unit8**

::

    node = onnx.helper.make_node(
        'BitShift',
        inputs=['x', 'y'],
        outputs=['z'],
        direction="RIGHT"
    )

    x = np.array([16, 4, 1]).astype(np.uint8)
    y = np.array([1, 2, 3]).astype(np.uint8)
    z = x >> y  # expected output [8, 1, 0]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_bitshift_right_uint8')

**right_unit16**

::

    node = onnx.helper.make_node(
        'BitShift',
        inputs=['x', 'y'],
        outputs=['z'],
        direction="RIGHT"
    )

    x = np.array([16, 4, 1]).astype(np.uint16)
    y = np.array([1, 2, 3]).astype(np.uint16)
    z = x >> y  # expected output [8, 1, 0]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_bitshift_right_uint16')

**right_unit32**

::

    node = onnx.helper.make_node(
        'BitShift',
        inputs=['x', 'y'],
        outputs=['z'],
        direction="RIGHT"
    )

    x = np.array([16, 4, 1]).astype(np.uint32)
    y = np.array([1, 2, 3]).astype(np.uint32)
    z = x >> y  # expected output [8, 1, 0]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_bitshift_right_uint32')

**right_unit64**

::

    node = onnx.helper.make_node(
        'BitShift',
        inputs=['x', 'y'],
        outputs=['z'],
        direction="RIGHT"
    )

    x = np.array([16, 4, 1]).astype(np.uint64)
    y = np.array([1, 2, 3]).astype(np.uint64)
    z = x >> y  # expected output [8, 1, 0]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_bitshift_right_uint64')

**left_unit8**

::

    node = onnx.helper.make_node(
        'BitShift',
        inputs=['x', 'y'],
        outputs=['z'],
        direction="LEFT"
    )

    x = np.array([16, 4, 1]).astype(np.uint8)
    y = np.array([1, 2, 3]).astype(np.uint8)
    z = x << y  # expected output [32, 16, 8]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_bitshift_left_uint8')

**left_unit16**

::

    node = onnx.helper.make_node(
        'BitShift',
        inputs=['x', 'y'],
        outputs=['z'],
        direction="LEFT"
    )

    x = np.array([16, 4, 1]).astype(np.uint16)
    y = np.array([1, 2, 3]).astype(np.uint16)
    z = x << y  # expected output [32, 16, 8]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_bitshift_left_uint16')

**left_unit32**

::

    node = onnx.helper.make_node(
        'BitShift',
        inputs=['x', 'y'],
        outputs=['z'],
        direction="LEFT"
    )

    x = np.array([16, 4, 1]).astype(np.uint32)
    y = np.array([1, 2, 3]).astype(np.uint32)
    z = x << y  # expected output [32, 16, 8]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_bitshift_left_uint32')

**left_unit64**

::

    node = onnx.helper.make_node(
        'BitShift',
        inputs=['x', 'y'],
        outputs=['z'],
        direction="LEFT"
    )

    x = np.array([16, 4, 1]).astype(np.uint64)
    y = np.array([1, 2, 3]).astype(np.uint64)
    z = x << y  # expected output [32, 16, 8]
    expect(node, inputs=[x, y], outputs=[z],
           name='test_bitshift_left_uint64')
