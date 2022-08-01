
.. _l-onnx-doccom.microsoft.nchwc-GlobalAveragePool:

=======================================
com.microsoft.nchwc - GlobalAveragePool
=======================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-nchwc-globalaveragepool-1:

GlobalAveragePool - 1 (com.microsoft.nchwc)
===========================================

**Version**

* **name**: `GlobalAveragePool (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.nchwc.GlobalAveragePool>`_
* **domain**: **com.microsoft.nchwc**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft.nchwc**.

**Summary**

For internal use.

**Inputs**

* **X** (heterogeneous) - **T**:

**Outputs**

* **Y** (heterogeneous) - **T**:

**Examples**

**globalaveragepool_precomputed**

::

    node = onnx.helper.make_node(
        'GlobalAveragePool',
        inputs=['x'],
        outputs=['y'],
    )
    x = np.array([[[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]]]).astype(np.float32)
    y = np.array([[[[5]]]]).astype(np.float32)
    expect(node, inputs=[x], outputs=[y], name='test_globalaveragepool_precomputed')
