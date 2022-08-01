
.. _l-onnx-docai.onnx.preview.training-Adagrad:

==================================
ai.onnx.preview.training - Adagrad
==================================

.. contents::
    :local:


.. _l-onnx-opai-onnx-preview-training-adagrad-1:

Adagrad - 1 (ai.onnx.preview.training)
======================================

**Version**

* **name**: `Adagrad (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#ai.onnx.preview.training.Adagrad>`_
* **domain**: **ai.onnx.preview.training**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1 of domain ai.onnx.preview.training**.

**Summary**

Compute one iteration of ADAGRAD, a stochastic gradient based optimization
algorithm. This operator can conduct the optimization of multiple tensor variables.

Let's define the behavior of this operator. As you can imagine, ADAGRAD requires
some parameters:

 - The initial learning-rate "R".
 - The update count "T". That is, the number of training iterations conducted.
 - A L2-norm regularization coefficient "norm_coefficient".
 - A learning-rate decay factor "decay_factor".
 - A small constant "epsilon" to avoid dividing-by-zero.

At each ADAGRAD iteration, the optimized tensors are moved along a direction
computed based on their estimated gradient and accumulated squared gradient. Assume
that only a single tensor "X" is updated by this operator. We need the value of "X",
its gradient "G", and its accumulated squared gradient "H". Therefore, variables in
this operator's input list are sequentially "R", "T", "X", "G", and "H". Other
parameters are given as attributes because they are usually constants. Also, the
corresponding output tensors are the new value of "X" (called "X_new"), and then
the new accumulated squared gradient (called "H_new"). Those outputs are computed
from the given inputs following the pseudo code below.

Let "+", "-", "*", and "/" are all element-wise arithmetic operations with
numpy-style broadcasting support. The pseudo code to compute those outputs is:

  // Compute a scalar learning-rate factor. At the first update of X, T is generally
  // 0 (0-based update index) or 1 (1-based update index).
  r = R / (1 + T * decay_factor);

  // Add gradient of 0.5 * norm_coefficient * ||X||_2^2, where ||X||_2 is the 2-norm.
  G_regularized = norm_coefficient * X + G;

  // Compute new accumulated squared gradient.
  H_new = H + G_regularized * G_regularized;

  // Compute the adaptive part of per-coordinate learning rate. Note that Sqrt(...)
  // computes element-wise square-root.
  H_adaptive = Sqrt(H_new) + epsilon

  // Compute the new value of "X".
  X_new = X - r * G_regularized / H_adaptive;

If one assign this operators to optimize multiple inputs, for example, "X_1" and "X_2", the same
pseudo code may be extended to handle all tensors jointly. More specifically, we can view "X" as a
concatenation of "X_1" and "X_2" (of course, their gradient and accumulate gradient should
be concatenated too) and then just reuse the entire pseudo code.

Note that ADAGRAD was first proposed in http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.
In that reference paper, this operator is a special case of the Figure 1's composite mirror
descent update.

**Attributes**

* **decay_factor**:
  The decay factor of learning rate after one update.The effective
  learning rate is computed by r = R / (1 + T * decay_factor). Default
  to 0 so that increasing update counts doesn't reduce the learning
  rate. Default value is ``0.0``.
* **epsilon**:
  Small scalar to avoid dividing by zero. Default value is ``9.999999974752427e-07``.
* **norm_coefficient**:
  Regularization coefficient in 0.5 * norm_coefficient * ||X||_2^2.
  Default to 0, which means no regularization. Default value is ``0.0``.

**Inputs**

Between 3 and 2147483647 inputs.

* **R** (heterogeneous) - **T1**:
  The initial learning rate.
* **T** (heterogeneous) - **T2**:
  The update count of "X". It should be a scalar.
* **inputs** (variadic) - **T3**:
  The current values of optimized tensors, followed by their
  respective gradients, followed by their respective accumulated
  squared gradients.For example, if two tensor "X_1" and "X_2" are
  optimized, The input list would be ["X_1", "X_2", gradient of "X_1",
  gradient of "X_2", accumulated squared gradient of "X_1",
  accumulated squared gradient of "X_2"].

**Outputs**

Between 1 and 2147483647 outputs.

* **outputs** (variadic) - **T3**:
  Updated values of optimized tensors, followed by their updated
  values of accumulated squared gradients. For example, if two tensor
  "X_1" and "X_2" are optimized, the output list would be [new value
  of "X_1," new value of "X_2" new accumulated squared gradient of
  "X_1", new accumulated squared gradient of "X_2"].

**Type Constraints**

* **T1** in (
  tensor(double),
  tensor(float)
  ):
  Constrain input types to float scalars.
* **T2** in (
  tensor(int64)
  ):
  Constrain input types to 64-bit integer scalars.
* **T3** in (
  tensor(double),
  tensor(float)
  ):
  Constrain input and output types to float tensors.

**Examples**

**adagrad**

::

    # Define operator attributes.
    norm_coefficient = 0.001
    epsilon = 1e-5
    decay_factor = 0.1

    # Create operator.
    node = onnx.helper.make_node('Adagrad',
                                 inputs=['R', 'T', 'X', 'G', 'H'],
                                 outputs=['X_new', 'H_new'],
                                 norm_coefficient=norm_coefficient,
                                 epsilon=epsilon,
                                 decay_factor=decay_factor,
                                 domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN
                                 )

    # Define operator inputs.
    r = np.array(0.1, dtype=np.float32)  # scalar
    t = np.array(0, dtype=np.int64)  # scalar
    x = np.array([1.0], dtype=np.float32)
    g = np.array([-1.0], dtype=np.float32)
    h = np.array([2.0], dtype=np.float32)

    # Compute expected outputs of Adagrad.
    x_new, h_new = apply_adagrad(r, t, x, g, h,
                                 norm_coefficient, epsilon, decay_factor)

    # Check results.
    expect(node, inputs=[r, t, x, g, h],
           outputs=[x_new, h_new], name='test_adagrad',
           opset_imports=[onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])

**adagrad_multiple**

::

    # Define operator attributes.
    norm_coefficient = 0.001
    epsilon = 1e-5
    decay_factor = 0.1

    node = onnx.helper.make_node('Adagrad',
                                 inputs=['R', 'T', 'X1', 'X2',
                                         'G1', 'G2', 'H1', 'H2'],
                                 outputs=['X1_new', 'X2_new',
                                          'H1_new', 'H2_new'],
                                 norm_coefficient=norm_coefficient,
                                 epsilon=epsilon,
                                 decay_factor=decay_factor,
                                 domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN
                                 )

    # Define operator inputs.
    r = np.array(0.1, dtype=np.float32)  # scalar
    t = np.array(0, dtype=np.int64)  # scalar

    x1 = np.array([1.0], dtype=np.float32)
    g1 = np.array([-1.0], dtype=np.float32)
    h1 = np.array([2.0], dtype=np.float32)

    x2 = np.array([1.0, 2.0], dtype=np.float32)
    g2 = np.array([-1.0, -3.0], dtype=np.float32)
    h2 = np.array([4.0, 1.0], dtype=np.float32)

    # Compute expected outputs of Adagrad.
    x1_new, h1_new = apply_adagrad(r, t, x1, g1, h1,
                                   norm_coefficient, epsilon, decay_factor)
    x2_new, h2_new = apply_adagrad(r, t, x2, g2, h2,
                                   norm_coefficient, epsilon, decay_factor)

    # Check results.
    expect(node, inputs=[r, t, x1, x2, g1, g2, h1, h2],
           outputs=[x1_new, x2_new, h1_new, h2_new], name='test_adagrad_multiple',
           opset_imports=[onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])
