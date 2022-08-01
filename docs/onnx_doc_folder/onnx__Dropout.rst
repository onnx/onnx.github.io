
.. _l-onnx-doc-Dropout:

=======
Dropout
=======

.. contents::
    :local:


.. _l-onnx-op-dropout-13:

Dropout - 13
============

**Version**

* **name**: `Dropout (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Dropout>`_
* **domain**: **main**
* **since_version**: **13**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 13**.

**Summary**

Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs,
output (floating-point tensor) and mask (optional `Tensor<bool>`). If `training_mode` is true then the output Y will be a random dropout;
Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode,
the user can simply not pass `training_mode` input or set it to false.
::

    output = scale * data * mask,

where
::

    scale = 1. / (1. - ratio).

This operator has **optional** inputs/outputs. See `ONNX <https://github.com/onnx/onnx/blob/master/docs/IR.md>`_ for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

**Attributes**

* **seed**:
  (Optional) Seed to the random generator, if not specified we will
  auto generate one.

**Inputs**

Between 1 and 3 inputs.

* **data** (heterogeneous) - **T**:
  The input data as Tensor.
* **ratio** (optional, heterogeneous) - **T1**:
  The ratio of random dropout, with value in [0, 1). If this input was
  not set, or if it was set to 0, the output would be a simple copy of
  the input. If it's non-zero, output will be a random dropout of the
  scaled input, which is typically the case during training. It is an
  optional value, if not specified it will default to 0.5.
* **training_mode** (optional, heterogeneous) - **T2**:
  If set to true then it indicates dropout is being used for training.
  It is an optional value hence unless specified explicitly, it is
  false. If it is false, ratio is ignored and the operation mimics
  inference mode where nothing will be dropped from the input data and
  if mask is requested as output it will contain all ones.

**Outputs**

Between 1 and 2 outputs.

* **output** (heterogeneous) - **T**:
  The output.
* **mask** (optional, heterogeneous) - **T2**:
  The output mask.

**Type Constraints**

* **T** in (
  tensor(bfloat16),
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
* **T1** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input 'ratio' types to float tensors.
* **T2** in (
  tensor(bool)
  ):
  Constrain output 'mask' types to boolean tensors.

**Examples**

**default**

::

    seed = np.int64(0)
    node = onnx.helper.make_node(
        'Dropout',
        inputs=['x'],
        outputs=['y'],
        seed=seed
    )

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = dropout(x)
    expect(node, inputs=[x], outputs=[y], name='test_dropout_default')

**default_ratio**

::

    seed = np.int64(0)
    node = onnx.helper.make_node(
        'Dropout',
        inputs=['x', 'r'],
        outputs=['y'],
        seed=seed
    )

    r = np.float32(0.1)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = dropout(x, r)
    expect(node, inputs=[x, r], outputs=[y], name='test_dropout_default_ratio')

**default_mask**

::

    seed = np.int64(0)
    node = onnx.helper.make_node(
        'Dropout',
        inputs=['x'],
        outputs=['y', 'z'],
        seed=seed
    )

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y, z = dropout(x, return_mask=True)
    expect(node, inputs=[x], outputs=[y, z], name='test_dropout_default_mask')

**default_mask_ratio**

::

        seed = np.int64(0)
        node = onnx.helper.make_node(
            'Dropout',
            inputs=['x', 'r'],
            outputs=['y', 'z'],
            seed=seed
        )

        r = np.float32(0.1)
        x = np.random.randn(3, 4, 5).astype(np.float32)
        y, z = dropout(x, r, return_mask=True)
        expect(node, inputs=[x, r], outputs=[y, z], name='test_dropout_default_mask_ratio')

    # Training tests.

**training_default**

::

    seed = np.int64(0)
    node = onnx.helper.make_node(
        'Dropout',
        inputs=['x', 'r', 't'],
        outputs=['y'],
        seed=seed
    )

    x = np.random.randn(3, 4, 5).astype(np.float32)
    r = np.float32(0.5)
    t = np.bool_(True)
    y = dropout(x, r, training_mode=t)
    expect(node, inputs=[x, r, t], outputs=[y], name='test_training_dropout_default')

**training_default_ratio_mask**

::

    seed = np.int64(0)
    node = onnx.helper.make_node(
        'Dropout',
        inputs=['x', 'r', 't'],
        outputs=['y', 'z'],
        seed=seed
    )

    x = np.random.randn(3, 4, 5).astype(np.float32)
    r = np.float32(0.5)
    t = np.bool_(True)
    y, z = dropout(x, r, training_mode=t, return_mask=True)
    expect(node, inputs=[x, r, t], outputs=[y, z], name='test_training_dropout_default_mask')

**training**

::

    seed = np.int64(0)
    node = onnx.helper.make_node(
        'Dropout',
        inputs=['x', 'r', 't'],
        outputs=['y'],
        seed=seed
    )

    x = np.random.randn(3, 4, 5).astype(np.float32)
    r = np.float32(0.75)
    t = np.bool_(True)
    y = dropout(x, r, training_mode=t)
    expect(node, inputs=[x, r, t], outputs=[y], name='test_training_dropout')

**training_ratio_mask**

::

    seed = np.int64(0)
    node = onnx.helper.make_node(
        'Dropout',
        inputs=['x', 'r', 't'],
        outputs=['y', 'z'],
        seed=seed
    )

    x = np.random.randn(3, 4, 5).astype(np.float32)
    r = np.float32(0.75)
    t = np.bool_(True)
    y, z = dropout(x, r, training_mode=t, return_mask=True)
    expect(node, inputs=[x, r, t], outputs=[y, z], name='test_training_dropout_mask')

**training_default_zero_ratio**

::

    seed = np.int64(0)
    node = onnx.helper.make_node(
        'Dropout',
        inputs=['x', 'r', 't'],
        outputs=['y'],
        seed=seed
    )

    x = np.random.randn(3, 4, 5).astype(np.float32)
    r = np.float32(0.0)
    t = np.bool_(True)
    y = dropout(x, r, training_mode=t)
    expect(node, inputs=[x, r, t], outputs=[y], name='test_training_dropout_zero_ratio')

**training_default_zero_ratio_mask**

::

        seed = np.int64(0)
        node = onnx.helper.make_node(
            'Dropout',
            inputs=['x', 'r', 't'],
            outputs=['y', 'z'],
            seed=seed
        )

        x = np.random.randn(3, 4, 5).astype(np.float32)
        r = np.float32(0.0)
        t = np.bool_(True)
        y, z = dropout(x, r, training_mode=t, return_mask=True)
        expect(node, inputs=[x, r, t], outputs=[y, z], name='test_training_dropout_zero_ratio_mask')

    # Old dropout tests

**default_old**

::

    node = onnx.helper.make_node(
        'Dropout',
        inputs=['x'],
        outputs=['y'],
    )

    x = np.array([-1, 0, 1]).astype(np.float32)
    y = x
    expect(node, inputs=[x], outputs=[y],
           name='test_dropout_default_old', opset_imports=[helper.make_opsetid("", 11)])

**random_old**

::

    node = onnx.helper.make_node(
        'Dropout',
        inputs=['x'],
        outputs=['y'],
        ratio=.2,
    )

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = x
    expect(node, inputs=[x], outputs=[y],
           name='test_dropout_random_old', opset_imports=[helper.make_opsetid("", 11)])

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">output (floating-point tensor) and mask (optional Tensor<bool>). If training_mode is true then the output Y will be a random dropout;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">output (floating-point tensor) and mask (optional Tensor<bool>). If training_mode is true then the output Y will be a random dropout;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the user can simply not pass training_mode input or set it to false.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the user can simply not pass training_mode input or set it to false.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    output = scale * data * mask,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    output = scale * data * mask,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">where</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">where</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">::</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    scale = 1. / (1. - ratio).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    scale = 1. / (1. - ratio).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator has **optional** inputs/outputs. See ONNX <https://github.com/onnx/onnx/blob/master/docs/IR.md>_ for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator has **optional** inputs/outputs. See ONNX <https://github.com/onnx/onnx/blob/master/docs/IR.md>_ for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **seed**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **seed**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (Optional) Seed to the random generator, if not specified we will</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (Optional) Seed to the random generator, if not specified we will</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  auto generate one.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  auto generate one.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 3 inputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 3 inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input data as Tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input data as Tensor.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **ratio** (optional, heterogeneous) - **T1**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **ratio** (optional, heterogeneous) - **T1**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The ratio of random dropout, with value in [0, 1). If this input was</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The ratio of random dropout, with value in [0, 1). If this input was</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  not set, or if it was set to 0, the output would be a simple copy of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  not set, or if it was set to 0, the output would be a simple copy of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the input. If it's non-zero, output will be a random dropout of the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the input. If it's non-zero, output will be a random dropout of the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  scaled input, which is typically the case during training. It is an</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  scaled input, which is typically the case during training. It is an</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  optional value, if not specified it will default to 0.5.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  optional value, if not specified it will default to 0.5.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **training_mode** (optional, heterogeneous) - **T2**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **training_mode** (optional, heterogeneous) - **T2**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  If set to true then it indicates dropout is being used for training.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  If set to true then it indicates dropout is being used for training.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  It is an optional value hence unless specified explicitly, it is</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  It is an optional value hence unless specified explicitly, it is</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  false. If it is false, ratio is ignored and the operation mimics</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  false. If it is false, ratio is ignored and the operation mimics</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  inference mode where nothing will be dropped from the input data and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  inference mode where nothing will be dropped from the input data and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  if mask is requested as output it will contain all ones.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  if mask is requested as output it will contain all ones.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2 outputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2 outputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **mask** (optional, heterogeneous) - **T2**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **mask** (optional, heterogeneous) - **T2**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output mask.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output mask.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">52</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(bfloat16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input 'ratio' types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input 'ratio' types to float tensors.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T2** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T2** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain output 'mask' types to boolean tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain output 'mask' types to boolean tensors.</code></td></tr>
    </table>

.. _l-onnx-op-dropout-12:

Dropout - 12
============

**Version**

* **name**: `Dropout (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Dropout>`_
* **domain**: **main**
* **since_version**: **12**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 12**.

**Summary**

Dropout takes an input floating-point tensor, an optional input ratio (floating-point scalar) and an optional input training_mode (boolean scalar). It produces two tensor outputs,
output (floating-point tensor) and mask (optional `Tensor<bool>`). If `training_mode` is true then the output Y will be a random dropout;
Note that this Dropout scales the masked input data by the following equation, so to convert the trained model into inference mode,
the user can simply not pass `training_mode` input or set it to false.
::

    output = scale * data * mask,

where
::

    scale = 1. / (1. - ratio).

This operator has **optional** inputs/outputs. See `ONNX <https://github.com/onnx/onnx/blob/master/docs/IR.md>`_ for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

**Attributes**

* **seed**:
  (Optional) Seed to the random generator, if not specified we will
  auto generate one.

**Inputs**

Between 1 and 3 inputs.

* **data** (heterogeneous) - **T**:
  The input data as Tensor.
* **ratio** (optional, heterogeneous) - **T1**:
  The ratio of random dropout, with value in [0, 1). If this input was
  not set, or if it was set to 0, the output would be a simple copy of
  the input. If it's non-zero, output will be a random dropout of the
  scaled input, which is typically the case during training. It is an
  optional value, if not specified it will default to 0.5.
* **training_mode** (optional, heterogeneous) - **T2**:
  If set to true then it indicates dropout is being used for training.
  It is an optional value hence unless specified explicitly, it is
  false. If it is false, ratio is ignored and the operation mimics
  inference mode where nothing will be dropped from the input data and
  if mask is requested as output it will contain all ones.

**Outputs**

Between 1 and 2 outputs.

* **output** (heterogeneous) - **T**:
  The output.
* **mask** (optional, heterogeneous) - **T2**:
  The output mask.

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
* **T1** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input 'ratio' types to float tensors.
* **T2** in (
  tensor(bool)
  ):
  Constrain output 'mask' types to boolean tensors.

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td><code>0</code></td><td><code>0</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">Dropout takes <span style="color:#BA4A00;">o</span>n<span style="color:#BA4A00;">e</span> input floating tensor and produces two tensor outputs,</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>Dropout takes <span style="color:#196F3D;">a</span>n input floating<span style="color:#196F3D;">-</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span> tensor<span style="color:#196F3D;">,</span> an<span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span>d <span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span>p<span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span>r<span style="color:#196F3D;">a</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">_</span><span style="color:#196F3D;">m</span>od<span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">I</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">d</span>uces two tensor outputs,</code></td></tr>
    <tr style="1px solid black;"><td><code>1</code></td><td><code>1</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">output (floating tensor) and mask (Tensor<bool>). <span style="color:#BA4A00;">D</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">e</span>n<span style="color:#BA4A00;">d</span>ing<span style="color:#BA4A00;"> </span>o<span style="color:#BA4A00;">n</span> <span style="color:#BA4A00;">w</span>hethe<span style="color:#BA4A00;">r</span> <span style="color:#BA4A00;">i</span>t i<span style="color:#BA4A00;">s</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>output (floating<span style="color:#196F3D;">-</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span> tensor) and mask (<span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;"> </span>Tensor<bool>). <span style="color:#196F3D;">I</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">i</span>ning<span style="color:#196F3D;">_</span><span style="color:#196F3D;">m</span>o<span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span> <span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span>he<span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span>the <span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span>t<span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span> <span style="color:#196F3D;">Y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">w</span>i<span style="color:#196F3D;">l</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">;</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">2</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">in test mode or not, the output Y will either be a random dropout, or a simple</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">3</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">copy of the input. Note that our implementation of Dropout does scaling in</code></td><td></td></tr>
    <tr style="1px solid black;"><td><code>4</code></td><td><code>2</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">t<span style="color:#BA4A00;">h</span>e t<span style="color:#BA4A00;">r</span>ai<span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">g</span> p<span style="color:#BA4A00;">h</span>ase<span style="color:#BA4A00;">,</span> s<span style="color:#BA4A00;">o</span> d<span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">r</span>ing <span style="color:#BA4A00;">t</span>e<span style="color:#BA4A00;">s</span>tin<span style="color:#BA4A00;">g</span> <span style="color:#BA4A00;">n</span>othin<span style="color:#BA4A00;">g</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">n</span>e<span style="color:#BA4A00;">e</span>d<span style="color:#BA4A00;">s</span> to <span style="color:#BA4A00;">b</span>e d<span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">n</span>e<span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">N</span><span style="color:#196F3D;">o</span>te t<span style="color:#196F3D;">h</span>a<span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span>i<span style="color:#196F3D;">s</span> <span style="color:#196F3D;">D</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span>p<span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">c</span>a<span style="color:#196F3D;">l</span><span style="color:#196F3D;">e</span>s<span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span>e <span style="color:#196F3D;">m</span><span style="color:#196F3D;">a</span>s<span style="color:#196F3D;">k</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span> <span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span>d<span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">w</span>ing e<span style="color:#196F3D;">q</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">a</span>ti<span style="color:#196F3D;">o</span>n<span style="color:#196F3D;">,</span> <span style="color:#196F3D;">s</span>o<span style="color:#196F3D;"> </span>t<span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span>h<span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span>ined <span style="color:#196F3D;">m</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span>to <span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">f</span>e<span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">e</span> <span style="color:#196F3D;">m</span><span style="color:#196F3D;">o</span>de<span style="color:#196F3D;">,</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">3</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">the user can simply not pass training_mode input or set it to false.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">4</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">::</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">5</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">6</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">    output = scale * data * mask,</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">7</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">8</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">where</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">9</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">::</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">10</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">11</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">    scale = 1. / (1. - ratio).</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">12</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator has **optional** inputs/outputs. See ONNX <https://github.com/onnx/onnx/blob/master/docs/IR.md>_ for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator has **optional** inputs/outputs. See ONNX <https://github.com/onnx/onnx/blob/master/docs/IR.md>_ for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">17</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **seed**:</code></td></tr>
    <tr style="1px solid black;"><td><code>9</code></td><td><code>18</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">*</span> <span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">*</span>ratio<span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">:</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code> <span style="color:#196F3D;"> </span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">O</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">S</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span>ra<span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span>t<span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span>i<span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">n</span>o<span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">w</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">w</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">l</span></code></td></tr>
    <tr style="1px solid black;"><td><code>10</code></td><td><code>19</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  <span style="color:#BA4A00;">T</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">r</span>at<span style="color:#BA4A00;">i</span>o <span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span>ra<span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">m</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">u</span>t<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">D</span>e<span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">t</span> <span style="color:#BA4A00;">v</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">u</span>e<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">0</span>.<span style="color:#BA4A00;">5</span><span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  a<span style="color:#196F3D;">u</span>to <span style="color:#196F3D;">g</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">e</span>rate <span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span>e.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">23</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">Between 1 and 3 inputs.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">24</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">26</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  The input data as Tensor.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">27</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **ratio** (optional, heterogeneous) - **T1**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">28</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  The ratio of random dropout, with value in [0, 1). If this input was</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">29</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  not set, or if it was set to 0, the output would be a simple copy of</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">30</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  the input. If it's non-zero, output will be a random dropout of the</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">31</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  scaled input, which is typically the case during training. It is an</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">32</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional value, if not specified it will default to 0.5.</code></td></tr>
    <tr style="1px solid black;"><td><code>15</code></td><td><code>33</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"> <span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">T</span><span style="color:#BA4A00;">h</span>e in<span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">t</span> <span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">a</span>t<span style="color:#BA4A00;">a</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span>s T<span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">*</span> <span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">_</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">d</span>e<span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span> <span style="color:#196F3D;">(</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span>i<span style="color:#196F3D;">o</span>n<span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">,</span> <span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span>t<span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span>s<span style="color:#196F3D;">)</span> <span style="color:#196F3D;">-</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span>T<span style="color:#196F3D;">2</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">:</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">34</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  If set to true then it indicates dropout is being used for training.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">35</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  It is an optional value hence unless specified explicitly, it is</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">36</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  false. If it is false, ratio is ignored and the operation mimics</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">37</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  inference mode where nothing will be dropped from the input data and</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">38</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  if mask is requested as output it will contain all ones.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2 outputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2 outputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output.</code></td></tr>
    <tr style="1px solid black;"><td><code>23</code></td><td><code>46</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">* **mask** (optional, heterogeneous) - **T<span style="color:#BA4A00;">1</span>**:</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>* **mask** (optional, heterogeneous) - **T<span style="color:#196F3D;">2</span>**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output mask.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output mask.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">58</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">59</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">60</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">61</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  ):</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">62</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Constrain input 'ratio' types to float tensors.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">63</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **T2** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td><code>37</code></td><td><code>66</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Constrain output mask types to boolean tensors.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Constrain output <span style="color:#196F3D;">'</span>mask<span style="color:#196F3D;">'</span> types to boolean tensors.</code></td></tr>
    </table>

.. _l-onnx-op-dropout-10:

Dropout - 10
============

**Version**

* **name**: `Dropout (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Dropout>`_
* **domain**: **main**
* **since_version**: **10**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 10**.

**Summary**

Dropout takes one input floating tensor and produces two tensor outputs,
output (floating tensor) and mask (`Tensor<bool>`). Depending on whether it is
in test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
This operator has **optional** inputs/outputs. See `ONNX <https://github.com/onnx/onnx/blob/master/docs/IR.md>`_ for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

**Attributes**

* **ratio**:
  The ratio of random dropout Default value is ``0.5``.

**Inputs**

* **data** (heterogeneous) - **T**:
  The input data as Tensor.

**Outputs**

Between 1 and 2 outputs.

* **output** (heterogeneous) - **T**:
  The output.
* **mask** (optional, heterogeneous) - **T1**:
  The output mask.

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
* **T1** in (
  tensor(bool)
  ):
  Constrain output mask types to boolean tensors.

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td><code>0</code></td><td><code>0</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">Dropout takes one input <span style="color:#BA4A00;">d</span>at<span style="color:#BA4A00;">a</span> <span style="color:#BA4A00;">(</span><span style="color:#BA4A00;">T</span>ensor<span style="color:#BA4A00;"><</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">></span><span style="color:#BA4A00;">)</span> and produces two <span style="color:#BA4A00;">T</span>ensor outputs,</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>Dropout takes one input <span style="color:#196F3D;">f</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span>at<span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span> <span style="color:#196F3D;">t</span>ensor and produces two <span style="color:#196F3D;">t</span>ensor outputs,</code></td></tr>
    <tr style="1px solid black;"><td><code>1</code></td><td><code>1</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">output (<span style="color:#BA4A00;">T</span>ensor<span style="color:#BA4A00;"><</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">></span>) and mask (Tensor<bool>). Depending on whether it is<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>output (<span style="color:#196F3D;">f</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span>ensor) and mask (Tensor<bool>). Depending on whether it is</code></td></tr>
    <tr style="1px solid black;"><td><code>2</code></td><td><code>2</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">test mode or not, the output Y will either be a random dropout, or a simple</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span>test mode or not, the output Y will either be a random dropout, or a simple</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">copy of the input. Note that our implementation of Dropout does scaling in</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">copy of the input. Note that our implementation of Dropout does scaling in</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the training phase, so during testing nothing needs to be done.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the training phase, so during testing nothing needs to be done.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator has **optional** inputs/outputs. See ONNX <https://github.com/onnx/onnx/blob/master/docs/IR.md>_ for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator has **optional** inputs/outputs. See ONNX <https://github.com/onnx/onnx/blob/master/docs/IR.md>_ for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **ratio**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **ratio**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The ratio of random dropout Default value is 0.5.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The ratio of random dropout Default value is 0.5.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input data as Tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input data as Tensor.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2 outputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2 outputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output.</code></td></tr>
    <tr style="1px solid black;"><td><code>23</code></td><td><code>23</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">* **mask** (optional, heterogeneous) - **T**:</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>* **mask** (optional, heterogeneous) - **T<span style="color:#196F3D;">1</span>**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output mask.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output mask.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">34</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **T1** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">35</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(bool)</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">36</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  ):</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">37</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Constrain output mask types to boolean tensors.</code></td></tr>
    </table>

.. _l-onnx-op-dropout-7:

Dropout - 7
===========

**Version**

* **name**: `Dropout (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Dropout>`_
* **domain**: **main**
* **since_version**: **7**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 7**.

**Summary**

Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,
output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in
test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.
This operator has **optional** inputs/outputs. See `ONNX <https://github.com/onnx/onnx/blob/master/docs/IR.md>`_ for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.

**Attributes**

* **ratio**:
  The ratio of random dropout Default value is ``0.5``.

**Inputs**

* **data** (heterogeneous) - **T**:
  The input data as Tensor.

**Outputs**

Between 1 and 2 outputs.

* **output** (heterogeneous) - **T**:
  The output.
* **mask** (optional, heterogeneous) - **T**:
  The output mask.

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">test mode or not, the output Y will either be a random dropout, or a simple</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">test mode or not, the output Y will either be a random dropout, or a simple</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">copy of the input. Note that our implementation of Dropout does scaling in</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">copy of the input. Note that our implementation of Dropout does scaling in</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the training phase, so during testing nothing needs to be done.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the training phase, so during testing nothing needs to be done.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">5</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;"></code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">6</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">**Attributes**</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">7</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;"></code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">8</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">* **is_test**:</code></td><td></td></tr>
    <tr style="1px solid black;"><td><code>9</code></td><td><code>5</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  <span style="color:#BA4A00;">(</span>int<span style="color:#BA4A00;">,</span> de<span style="color:#BA4A00;">f</span>au<span style="color:#BA4A00;">l</span>t <span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">)</span> if <span style="color:#BA4A00;">n</span>on<span style="color:#BA4A00;">z</span>ero<span style="color:#BA4A00;">,</span> run dr<span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">o</span>ut in test<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">m</span>o<span style="color:#BA4A00;">d</span>e w<span style="color:#BA4A00;">h</span>ere the</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">T</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span> <span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span> <span style="color:#196F3D;">h</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span>i<span style="color:#196F3D;">o</span>n<span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span>t<span style="color:#196F3D;">s</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">.</span> <span style="color:#196F3D;">S</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">O</span><span style="color:#196F3D;">N</span><span style="color:#196F3D;">N</span><span style="color:#196F3D;">X</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"><</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">:</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">x</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">/</span>d<span style="color:#196F3D;">o</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">I</span><span style="color:#196F3D;">R</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">></span><span style="color:#196F3D;">_</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span>e<span style="color:#196F3D;"> </span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">t</span>a<span style="color:#196F3D;">i</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">o</span>ut <span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span> <span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span>i<span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span>f o<span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span>n<span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span>e<span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">A</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">t</span>r<span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span>o<span style="color:#196F3D;">f</span> <span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span>r<span style="color:#196F3D;">g</span>u<span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span>n<span style="color:#196F3D;">t</span><span style="color:#196F3D;">'</span><span style="color:#196F3D;">s</span> <span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span>d<span style="color:#196F3D;">i</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span>r<span style="color:#196F3D;">g</span>u<span style="color:#196F3D;">m</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span>t<span style="color:#196F3D;">.</span> <span style="color:#196F3D;">T</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span>i<span style="color:#196F3D;">l</span><span style="color:#196F3D;">i</span>n<span style="color:#196F3D;">g</span> <span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span>t<span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span>e<span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span>s<span style="color:#196F3D;"> </span><span style="color:#196F3D;">(</span>t<span style="color:#196F3D;">h</span>o<span style="color:#196F3D;">s</span>e <span style="color:#196F3D;">n</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span>we<span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span>r<span style="color:#196F3D;">g</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span>e<span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span> th<span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">r</span>e<span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">6</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td><code>10</code></td><td><code>7</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">u</span>t<span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">u</span>t<span style="color:#BA4A00;"> </span>i<span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">y</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">Y</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">=</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">X</span><span style="color:#BA4A00;">.</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">D</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">a</span>u<span style="color:#BA4A00;">l</span>t<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">v</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">u</span>e<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span>s<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span><span style="color:#196F3D;">A</span>tt<span style="color:#196F3D;">r</span>i<span style="color:#196F3D;">b</span>utes<span style="color:#196F3D;">*</span><span style="color:#196F3D;">*</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">8</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **ratio**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **ratio**:</code></td></tr>
    <tr style="1px solid black;"><td><code>12</code></td><td><code>10</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  <span style="color:#BA4A00;">(</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">0</span><span style="color:#BA4A00;">.</span><span style="color:#BA4A00;">5</span><span style="color:#BA4A00;">)</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span>he ratio of random dropout Default value is 0.5.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  <span style="color:#196F3D;">T</span>he ratio of random dropout Default value is 0.5.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input data as Tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input data as Tensor.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2 outputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2 outputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **mask** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **mask** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td><code>26</code></td><td><code>24</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  The output mask.<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">I</span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">_</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">z</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">,</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">f</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  The output mask.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    </table>

.. _l-onnx-op-dropout-6:

Dropout - 6
===========

**Version**

* **name**: `Dropout (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Dropout>`_
* **domain**: **main**
* **since_version**: **6**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 6**.

**Summary**

Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,
output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in
test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.

**Attributes**

* **is_test**:
  (int, default 0) if nonzero, run dropout in test mode where the
  output is simply Y = X. Default value is ``0``.
* **ratio**:
  (float, default 0.5) the ratio of random dropout Default value is ``0.5``.

**Inputs**

* **data** (heterogeneous) - **T**:
  The input data as Tensor.

**Outputs**

Between 1 and 2 outputs.

* **output** (heterogeneous) - **T**:
  The output.
* **mask** (optional, heterogeneous) - **T**:
  The output mask. If is_test is nonzero, this output is not filled.

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">test mode or not, the output Y will either be a random dropout, or a simple</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">test mode or not, the output Y will either be a random dropout, or a simple</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">copy of the input. Note that our implementation of Dropout does scaling in</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">copy of the input. Note that our implementation of Dropout does scaling in</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the training phase, so during testing nothing needs to be done.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the training phase, so during testing nothing needs to be done.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">8</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">* **consumed_inputs**:</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">9</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">  legacy optimization attribute.</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **is_test**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **is_test**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (int, default 0) if nonzero, run dropout in test mode where the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (int, default 0) if nonzero, run dropout in test mode where the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output is simply Y = X. Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output is simply Y = X. Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **ratio**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **ratio**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (float, default 0.5) the ratio of random dropout Default value is 0.5.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (float, default 0.5) the ratio of random dropout Default value is 0.5.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **data** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input data as Tensor.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input data as Tensor.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2 outputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2 outputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **mask** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **mask** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output mask. If is_test is nonzero, this output is not filled.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output mask. If is_test is nonzero, this output is not filled.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    </table>

.. _l-onnx-op-dropout-1:

Dropout - 1
===========

**Version**

* **name**: `Dropout (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Dropout>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: False

This version of the operator has been available
**since version 1**.

**Summary**

Dropout takes one input data (Tensor<float>) and produces two Tensor outputs,
output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in
test mode or not, the output Y will either be a random dropout, or a simple
copy of the input. Note that our implementation of Dropout does scaling in
the training phase, so during testing nothing needs to be done.

**Attributes**

* **consumed_inputs**:
  legacy optimization attribute.
* **is_test**:
  (int, default 0) if nonzero, run dropout in test mode where the
  output is simply Y = X. Default value is ``0``.
* **ratio**:
  (float, default 0.5) the ratio of random dropout Default value is ``0.5``.

**Inputs**

* **data** (heterogeneous) - **T**:
  The input data as Tensor.

**Outputs**

Between 1 and 2 outputs.

* **output** (heterogeneous) - **T**:
  The output.
* **mask** (optional, heterogeneous) - **T**:
  The output mask. If is_test is nonzero, this output is not filled.

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
