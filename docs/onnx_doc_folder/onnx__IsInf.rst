
.. _l-onnx-doc-IsInf:

=====
IsInf
=====

.. contents::
    :local:


.. _l-onnx-op-isinf-10:

IsInf - 10
==========

**Version**

* **name**: `IsInf (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#IsInf>`_
* **domain**: **main**
* **since_version**: **10**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 10**.

**Summary**

Map infinity to true and other values to false.

**Attributes**

* **detect_negative**:
  (Optional) Whether map negative infinity to true. Default to 1 so
  that negative infinity induces true. Set this attribute to 0 if
  negative infinity should be mapped to false. Default value is ``1``.
* **detect_positive**:
  (Optional) Whether map positive infinity to true. Default to 1 so
  that positive infinity induces true. Set this attribute to 0 if
  positive infinity should be mapped to false. Default value is ``1``.

**Inputs**

* **X** (heterogeneous) - **T1**:
  input

**Outputs**

* **Y** (heterogeneous) - **T2**:
  output

**Type Constraints**

* **T1** in (
  tensor(double),
  tensor(float)
  ):
  Constrain input types to float tensors.
* **T2** in (
  tensor(bool)
  ):
  Constrain output types to boolean tensors.

**Examples**

**infinity**

::

    node = onnx.helper.make_node('IsInf',
                                 inputs=['x'],
                                 outputs=['y'],
                                 )

    x = np.array([-1.2, np.nan, np.inf, 2.8, np.NINF, np.inf],
                 dtype=np.float32)
    y = np.isinf(x)
    expect(node, inputs=[x], outputs=[y], name='test_isinf')

**positive_infinity_only**

::

    node = onnx.helper.make_node('IsInf',
                                 inputs=['x'],
                                 outputs=['y'],
                                 detect_negative=0
                                 )

    x = np.array([-1.7, np.nan, np.inf, 3.6, np.NINF, np.inf],
                 dtype=np.float32)
    y = np.isposinf(x)
    expect(node, inputs=[x], outputs=[y], name='test_isinf_positive')

**negative_infinity_only**

::

    node = onnx.helper.make_node('IsInf',
                                 inputs=['x'],
                                 outputs=['y'],
                                 detect_positive=0
                                 )

    x = np.array([-1.7, np.nan, np.inf, -3.6, np.NINF, np.inf],
                 dtype=np.float32)
    y = np.isneginf(x)
    expect(node, inputs=[x], outputs=[y], name='test_isinf_negative')
