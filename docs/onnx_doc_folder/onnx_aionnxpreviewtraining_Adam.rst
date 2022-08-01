
.. _l-onnx-docai.onnx.preview.training-Adam:

===============================
ai.onnx.preview.training - Adam
===============================

.. contents::
    :local:


.. _l-onnx-opai-onnx-preview-training-adam-1:

Adam - 1 (ai.onnx.preview.training)
===================================

**Version**

* **name**: `Adam (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#ai.onnx.preview.training.Adam>`_
* **domain**: **ai.onnx.preview.training**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1 of domain ai.onnx.preview.training**.

**Summary**

Compute one iteration of Adam, a stochastic gradient based optimization
algorithm. This operator can conduct the optimization of multiple tensor variables.

Let's define the behavior of this operator. First of all, Adam requires
some parameters:

 - The learning-rate "R".
 - The update count "T". That is, the number of training iterations conducted.
 - A L2-norm regularization coefficient "norm_coefficient".
 - A small constant "epsilon" to avoid dividing-by-zero.
 - Two coefficients, "alpha" and "beta".

At each Adam iteration, the optimized tensors are moved along a direction
computed based on their exponentially-averaged historical gradient and
exponentially-averaged historical squared gradient. Assume that only a tensor
"X" is being optimized. The rest of required information is

 - the value of "X",
 - "X"'s gradient (denoted by "G"),
 - "X"'s exponentially-averaged historical gradient (denoted by "V"), and
 - "X"'s exponentially-averaged historical squared gradient (denoted by "H").

Some of those parameters are passed into this operator as input tensors and others
are stored as this operator's attributes. Specifically, this operator's input tensor
list is ["R", "T", "X", "G", "V", "H"]. That is, "R" is the first input, "T" is
the second input, and so on. Other parameters are given as attributes because they
are constants. Moreover, the corresponding output tensors are

 - the new value of "X" (called "X_new"),
 - the new exponentially-averaged historical gradient (denoted by "V_new"), and
 - the new exponentially-averaged historical squared gradient (denoted by "H_new").

Those outputs are computed following the pseudo code below.

Let "+", "-", "*", and "/" are all element-wise arithmetic operations with
numpy-style broadcasting support. The pseudo code to compute those outputs is:

  // Add gradient of 0.5 * norm_coefficient * ||X||_2^2, where ||X||_2 is the 2-norm.
  G_regularized = norm_coefficient * X + G

  // Update exponentially-averaged historical gradient.
  V_new = alpha * V + (1 - alpha) * G_regularized

  // Update exponentially-averaged historical squared gradient.
  H_new = beta * H + (1 - beta) * G_regularized * G_regularized

  // Compute the element-wise square-root of H_new. V_new will be element-wisely
  // divided by H_sqrt for a better update direction.
  H_sqrt = Sqrt(H_new) + epsilon

  // Compute learning-rate. Note that "alpha**T"/"beta**T" is alpha's/beta's T-th power.
  R_adjusted = T > 0 ? R * Sqrt(1 - beta**T) / (1 - alpha**T) : R

  // Compute new value of "X".
  X_new = X - R_adjusted * V_new / H_sqrt

  // Post-update regularization.
  X_final = (1 - norm_coefficient_post) * X_new

If there are multiple inputs to be optimized, the pseudo code will be applied
independently to each of them.

**Attributes**

* **alpha**:
  Coefficient of previously accumulated gradient in running average.
  Default to 0.9. Default value is ``0.8999999761581421``.
* **beta**:
  Coefficient of previously accumulated squared-gradient in running
  average. Default to 0.999. Default value is ``0.9990000128746033``.
* **epsilon**:
  Small scalar to avoid dividing by zero. Default value is ``9.999999974752427e-07``.
* **norm_coefficient**:
  Regularization coefficient of 0.5 * norm_coefficient * ||X||_2^2.
  Default to 0, which means no regularization. Default value is ``0.0``.
* **norm_coefficient_post**:
  Regularization coefficient of 0.5 * norm_coefficient * ||X||_2^2.
  Default to 0, which means no regularization. Default value is ``0.0``.

**Inputs**

Between 3 and 2147483647 inputs.

* **R** (heterogeneous) - **T1**:
  The initial learning rate.
* **T** (heterogeneous) - **T2**:
  The update count of "X". It should be a scalar.
* **inputs** (variadic) - **T3**:
  The tensors to be optimized, followed by their respective gradients,
  followed by their respective accumulated gradients (aka momentum),
  followed by their respective accumulated squared gradients. For
  example, to optimize tensors "X_1" and "X_2,", the input list would
  be ["X_1", "X_2", gradient of "X_1", gradient of "X_2", accumulated
  gradient of "X_1", accumulated gradient of "X_2", accumulated
  squared gradient of "X_1", accumulated squared gradient of "X_2"].

**Outputs**

Between 1 and 2147483647 outputs.

* **outputs** (variadic) - **T3**:
  New values of optimized tensors, followed by their respective new
  accumulated gradients, followed by their respective new accumulated
  squared gradients. For example, if two tensors "X_1" and "X_2" are
  optimized, the outputs list would be [new value of "X_1", new value
  of "X_2", new accumulated gradient of "X_1", new accumulated
  gradient of "X_2", new accumulated squared gradient of "X_1", new
  accumulated squared gradient of "X_2"].

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

**adam**

::

    # Define operator attributes.
    norm_coefficient = 0.001
    alpha = 0.95
    beta = 0.1
    epsilon = 1e-7

    # Create operator.
    node = onnx.helper.make_node('Adam',
                                 inputs=['R', 'T', 'X', 'G', 'V', 'H'],
                                 outputs=['X_new', 'V_new', 'H_new'],
                                 norm_coefficient=norm_coefficient,
                                 alpha=alpha,
                                 beta=beta,
                                 epsilon=epsilon,
                                 domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN
                                 )

    # Define operator inputs.
    r = np.array(0.1, dtype=np.float32)  # scalar
    t = np.array(0, dtype=np.int64)  # scalar
    x = np.array([1.2, 2.8], dtype=np.float32)
    g = np.array([-0.94, -2.5], dtype=np.float32)
    v = np.array([1.7, 3.6], dtype=np.float32)
    h = np.array([0.1, 0.1], dtype=np.float32)

    # Compute expected outputs of Adam.
    x_new, v_new, h_new = apply_adam(r, t, x, g, v, h,
                                     norm_coefficient, 0.0, alpha, beta,
                                     epsilon)

    # Check results.
    expect(node, inputs=[r, t, x, g, v, h],
           outputs=[x_new, v_new, h_new], name='test_adam',
           opset_imports=[onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])

**adam_multiple**

::

    # Define operator attributes.
    norm_coefficient = 0.001
    alpha = 0.95
    beta = 0.85
    epsilon = 1e-2

    node = onnx.helper.make_node('Adam',
                                 inputs=['R', 'T', 'X1', 'X2',
                                         'G1', 'G2', 'V1', 'V2',
                                         'H1', 'H2'],
                                 outputs=['X1_new', 'X2_new',
                                          'V1_new', 'V2_new',
                                          'H1_new', 'H2_new'],
                                 norm_coefficient=norm_coefficient,
                                 alpha=alpha,
                                 beta=beta,
                                 domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN
                                 )

    # Define operator inputs.
    r = np.array(0.1, dtype=np.float32)  # scalar
    t = np.array(0, dtype=np.int64)  # scalar

    x1 = np.array([1.0], dtype=np.float32)
    g1 = np.array([-1.0], dtype=np.float32)
    v1 = np.array([2.0], dtype=np.float32)
    h1 = np.array([0.5], dtype=np.float32)

    x2 = np.array([1.0, 2.0], dtype=np.float32)
    g2 = np.array([-1.0, -3.0], dtype=np.float32)
    v2 = np.array([4.0, 1.0], dtype=np.float32)
    h2 = np.array([1.0, 10.0], dtype=np.float32)

    # Compute expected outputs of Adam.
    x1_new, v1_new, h1_new = apply_adam(r, t, x1, g1, v1, h1,
                                norm_coefficient, 0.0, alpha, beta,
                                epsilon)
    x2_new, v2_new, h2_new = apply_adam(r, t, x2, g2, v2, h2,
                                norm_coefficient, 0.0, alpha, beta,
                                epsilon)

    # Check results.
    expect(node, inputs=[r, t, x1, x2, g1, g2, v1, v2, h1, h2],
           outputs=[x1_new, x2_new, v1_new, v2_new, h1_new, h2_new],
           name='test_adam_multiple',
           opset_imports=[onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])
