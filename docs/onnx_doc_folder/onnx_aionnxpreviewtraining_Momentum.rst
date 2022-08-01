
.. _l-onnx-docai.onnx.preview.training-Momentum:

===================================
ai.onnx.preview.training - Momentum
===================================

.. contents::
    :local:


.. _l-onnx-opai-onnx-preview-training-momentum-1:

Momentum - 1 (ai.onnx.preview.training)
=======================================

**Version**

* **name**: `Momentum (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#ai.onnx.preview.training.Momentum>`_
* **domain**: **ai.onnx.preview.training**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1 of domain ai.onnx.preview.training**.

**Summary**

Compute one iteration of stochastic gradient update with momentum.
This operator can conduct the optimization of multiple tensor variables.

Let's define the behavior of this operator. As you can imagine, SG with momentum requires
several parameters:

 - The learning-rate "R".
 - The update count "T". That is, the number of conducted training iterations. It should
   be zero in the first training iteration.
 - A L2-norm regularization coefficient "norm_coefficient".
 - A decay coefficient of previous accumulated gradient (i.e., momentum) "alpha".
 - The scaling coefficient of current gradient "beta".
 - An attribute to choose either standard momentum or Nesterov's momentum "mode" should
   be used.

For the sake of simplicity, assume that there is only one tensor (called "X") to be optimized.
Other necessary inputs are "X"'s gradient (called "G") and "X"'s momentum (called "V"). This
Momentum operator maps all these inputs to the new value of "X" (called "X_new") and its new
momentum (called "V_new").

This operator supports two different momentum algorithms. Set the attribute "mode" to
"nesterov" if Nesterov's momentum is desired. Otherwise, set the attribute "model" to
"standard" to use standard momentum. Computation details are described subsequently.

Let "+", "-", "*", and "/" are all element-wise operations with numpy-style broadcasting.

Pseudo code for SG with standard momentum:

  // Add gradient of 0.5 * norm_coefficient * ||X||^2, where ||X|| is the sum of squared
  // values of all elements in X.
  G_regularized = norm_coefficient * X + G

  // In the first training iteration, beta should always be 1.
  beta_adjusted = T > 0 ? beta : 1

  // Compute the current momentum based on previous momentum and the current gradient.
  V_new = alpha * V + beta_adjusted * G_regularized

  // Update X.
  X_new = X - R * V_new

Pseudo code for SG with Nesterov's momentum:

  // Add gradient of 0.5 * norm_coefficient * ||X||^2, where ||X|| is the sum of squared
  // values of all elements in X.
  G_regularized = norm_coefficient * X + G;

  // In the first training iteration, beta should always be 1.
  beta_adjusted = T > 0 ? beta : 1

  // Compute the current momentum based on previous momentum and the current gradient.
  V_new = alpha * V + beta_adjusted * G_regularized;

  // Compute final update direction and then update X.
  X_new = X - R * (G_regularized + alpha * V_new)

If one assign this operators to optimize multiple inputs, for example, "X_1" and "X_2". The same
pseudo code would be extended to handle all tensors jointly. More specifically, we can view "X" as a
concatenation of "X_1" and "X_2" (of course, their gradient and accumulate gradient should
be concatenated too) and then our pseudo code becomes applicable.

**Attributes**

* **alpha** (required):
  The decay factor of momentum. It should be a scalar.
* **beta** (required):
  The coefficient of gradient in computing new momentum. It should be
  a scalar.
* **mode** (required):
  Its value should be either "nesterov" or "standard". The value
  "nesterov" leads to the use of Nesterov's momentum while "standard"
  invokes stochastic gradient method using standard momentum
* **norm_coefficient** (required):
  Coefficient of 0.5 * norm_coefficient * ||X||^2.

**Inputs**

Between 3 and 2147483647 inputs.

* **R** (heterogeneous) - **T1**:
  The learning rate.
* **T** (heterogeneous) - **T2**:
  Update count of "X". It should be a scalar.
* **inputs** (variadic) - **T3**:
  It sequentially contains the current values of optimized tensors,
  then their gradient tensors, and finally their momentum tensors. For
  example, if two tensors "X_1" and "X_2" are optimized, The expected
  input list would be ["X_1", "X_2", gradient of "X_1", gradient of
  "X_2", momentum of "X_1", momentum of "X_2"].

**Outputs**

Between 1 and 2147483647 outputs.

* **outputs** (variadic) - **T3**:
  It sequentially contains the new values of optimized tensors and
  then the new values of their momentum tensors. For example, if two
  tensors "X_1" and "X_2" are optimized, the output list would be [new
  value of "X_1," new value of "X_2" new momentum of "X_1", new
  momentum of "X_2"].

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
  Constrain input types to float tensors.

**Examples**

**momentum**

::

    # Define operator attributes.
    norm_coefficient = 0.001
    alpha = 0.95
    beta = 0.1

    # Create operator.
    node = onnx.helper.make_node('Momentum',
                                 inputs=['R', 'T', 'X', 'G', 'V'],
                                 outputs=['X_new', 'V_new'],
                                 norm_coefficient=norm_coefficient,
                                 alpha=alpha,
                                 beta=beta,
                                 mode='standard',
                                 domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN
                                 )

    # Define operator inputs.
    r = np.array(0.1, dtype=np.float32)  # scalar
    t = np.array(0, dtype=np.int64)  # scalar
    x = np.array([1.2, 2.8], dtype=np.float32)
    g = np.array([-0.94, -2.5], dtype=np.float32)
    v = np.array([1.7, 3.6], dtype=np.float32)

    # Compute expected outputs of Momentum.
    x_new, v_new = apply_momentum(r, t, x, g, v,
                                  norm_coefficient, alpha, beta)

    # Check results.
    expect(node, inputs=[r, t, x, g, v],
           outputs=[x_new, v_new], name='test_momentum',
           opset_imports=[onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])

**nesterov_momentum**

::

    # Define operator attributes.
    norm_coefficient = 0.01
    alpha = 0.95
    beta = 1.0

    # Create operator.
    node = onnx.helper.make_node('Momentum',
                                 inputs=['R', 'T', 'X', 'G', 'V'],
                                 outputs=['X_new', 'V_new'],
                                 norm_coefficient=norm_coefficient,
                                 alpha=alpha,
                                 beta=beta,
                                 mode='nesterov',
                                 domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN
                                 )

    # Define operator inputs.
    r = np.array(0.1, dtype=np.float32)  # scalar
    t = np.array(0, dtype=np.int64)  # scalar
    x = np.array([1.2, 2.8], dtype=np.float32)
    g = np.array([-0.94, -2.5], dtype=np.float32)
    v = np.array([1.7, 3.6], dtype=np.float32)

    # Compute expected outputs of Momentum.
    x_new, v_new = apply_nesterov(r, t, x, g, v,
                                  norm_coefficient, alpha, beta)

    # Check results.
    expect(node, inputs=[r, t, x, g, v],
           outputs=[x_new, v_new], name='test_nesterov_momentum',
           opset_imports=[onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])

**momentum_multiple**

::

    # Define operator attributes.
    norm_coefficient = 0.001
    alpha = 0.95
    beta = 0.85

    node = onnx.helper.make_node('Momentum',
                                 inputs=['R', 'T', 'X1', 'X2',
                                         'G1', 'G2', 'H1', 'H2'],
                                 outputs=['X1_new', 'X2_new',
                                          'V1_new', 'V2_new'],
                                 norm_coefficient=norm_coefficient,
                                 alpha=alpha,
                                 beta=beta,
                                 mode='standard',
                                 domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN
                                 )

    # Define operator inputs.
    r = np.array(0.1, dtype=np.float32)  # scalar
    t = np.array(0, dtype=np.int64)  # scalar

    x1 = np.array([1.0], dtype=np.float32)
    g1 = np.array([-1.0], dtype=np.float32)
    v1 = np.array([2.0], dtype=np.float32)

    x2 = np.array([1.0, 2.0], dtype=np.float32)
    g2 = np.array([-1.0, -3.0], dtype=np.float32)
    v2 = np.array([4.0, 1.0], dtype=np.float32)

    # Compute expected outputs of Momentum.
    x1_new, v1_new = apply_momentum(r, t, x1, g1, v1,
                                    norm_coefficient, alpha, beta)
    x2_new, v2_new = apply_momentum(r, t, x2, g2, v2,
                                    norm_coefficient, alpha, beta)

    # Check results.
    expect(node, inputs=[r, t, x1, x2, g1, g2, v1, v2],
           outputs=[x1_new, x2_new, v1_new, v2_new], name='test_momentum_multiple',
           opset_imports=[onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])
