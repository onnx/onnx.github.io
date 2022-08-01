
.. _l-onnx-doccom.microsoft.nchwc-GlobalMaxPool:

===================================
com.microsoft.nchwc - GlobalMaxPool
===================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-nchwc-globalmaxpool-1:

GlobalMaxPool - 1 (com.microsoft.nchwc)
=======================================

**Version**

* **name**: `GlobalMaxPool (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.nchwc.GlobalMaxPool>`_
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
