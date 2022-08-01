
.. _l-onnx-doc-ThresholdedRelu:

===============
ThresholdedRelu
===============

.. contents::
    :local:


.. _l-onnx-op-thresholdedrelu-10:

ThresholdedRelu - 10
====================

**Version**

* **name**: `ThresholdedRelu (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#ThresholdedRelu>`_
* **domain**: **main**
* **since_version**: **10**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 10**.

**Summary**

ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
is applied to the tensor elementwise.

**Attributes**

* **alpha**:
  Threshold value Default value is ``1.0``.

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
  Constrain input and output types to float tensors.

**Examples**

**default**

::

    default_alpha = 1.0
    node = onnx.helper.make_node(
        'ThresholdedRelu',
        inputs=['x'],
        outputs=['y']
    )
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.clip(x, default_alpha, np.inf)
    y[y == default_alpha] = 0

    expect(node, inputs=[x], outputs=[y],
           name='test_thresholdedrelu_default')
