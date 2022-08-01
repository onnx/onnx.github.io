
.. _l-onnx-doc-Bernoulli:

=========
Bernoulli
=========

.. contents::
    :local:


.. _l-onnx-op-bernoulli-15:

Bernoulli - 15
==============

**Version**

* **name**: `Bernoulli (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Bernoulli>`_
* **domain**: **main**
* **since_version**: **15**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 15**.

**Summary**

Draws binary random numbers (0 or 1) from a Bernoulli distribution. The input tensor should be a tensor
containing probabilities p (a value in the range [0,1]) to be used for drawing the binary random number,
where an output of 1 is produced with probability p and an output of 0 is produced with probability (1-p).

This operator is non-deterministic and may not produce the same values in different
implementations (even if a seed is specified).

**Attributes**

* **dtype**:
  The data type for the elements of the output tensor. if not
  specified, we will use the data type of the input tensor.
* **seed**:
  (Optional) Seed to the random generator, if not specified we will
  auto generate one.

**Inputs**

* **input** (heterogeneous) - **T1**:
  All values in input have to be in the range:[0, 1].

**Outputs**

* **output** (heterogeneous) - **T2**:
  The returned output tensor only has values 0 or 1, same shape as
  input tensor.

**Type Constraints**

* **T1** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input types to float tensors.
* **T2** in (
  tensor(bfloat16),
  tensor(bool),
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
  Constrain output types to all numeric tensors and bool tensors.

**Examples**

**bernoulli_without_dtype**

::

    node = onnx.helper.make_node(
        'Bernoulli',
        inputs=['x'],
        outputs=['y'],
    )

    x = np.random.uniform(0.0, 1.0, 10).astype(np.float)
    y = bernoulli_reference_implementation(x, np.float)
    expect(node, inputs=[x], outputs=[y], name='test_bernoulli')

**bernoulli_with_dtype**

::

    node = onnx.helper.make_node(
        'Bernoulli',
        inputs=['x'],
        outputs=['y'],
        dtype=onnx.TensorProto.DOUBLE,
    )

    x = np.random.uniform(0.0, 1.0, 10).astype(np.float32)
    y = bernoulli_reference_implementation(x, np.float64)
    expect(node, inputs=[x], outputs=[y], name='test_bernoulli_double')

**bernoulli_with_seed**

::

    seed = np.float(0)
    node = onnx.helper.make_node(
        'Bernoulli',
        inputs=['x'],
        outputs=['y'],
        seed=seed,
    )

    x = np.random.uniform(0.0, 1.0, 10).astype(np.float32)
    y = bernoulli_reference_implementation(x, np.float32)
    expect(node, inputs=[x], outputs=[y], name='test_bernoulli_seed')
