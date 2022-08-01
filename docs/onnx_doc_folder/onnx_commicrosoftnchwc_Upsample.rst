
.. _l-onnx-doccom.microsoft.nchwc-Upsample:

==============================
com.microsoft.nchwc - Upsample
==============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-nchwc-upsample-1:

Upsample - 1 (com.microsoft.nchwc)
==================================

**Version**

* **name**: `Upsample (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.nchwc.Upsample>`_
* **domain**: **com.microsoft.nchwc**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft.nchwc**.

**Summary**

For internal use.

**Attributes**

* **coordinate_transformation_mode**:
 Default value is ``?``.
* **mode**:
 Default value is ``?``.
* **scales**:
 Default value is ``?``.

**Inputs**

* **X** (heterogeneous) - **T**:

**Outputs**

* **Y** (heterogeneous) - **T**:

**Examples**

**nearest**

::

    node = onnx.helper.make_node(
        'Upsample',
        inputs=['X', 'scales'],
        outputs=['Y'],
        mode='nearest',
    )

    data = np.array([[[
        [1, 2],
        [3, 4],
    ]]], dtype=np.float32)

    scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)

    output = np.array([[[
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [3, 3, 3, 4, 4, 4],
        [3, 3, 3, 4, 4, 4],
    ]]], dtype=np.float32)

    expect(node, inputs=[data, scales], outputs=[output],
           name='test_upsample_nearest', opset_imports=[helper.make_opsetid("", 9)])
