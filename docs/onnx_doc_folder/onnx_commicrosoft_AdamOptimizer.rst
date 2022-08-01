
.. _l-onnx-doccom.microsoft-AdamOptimizer:

=============================
com.microsoft - AdamOptimizer
=============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-adamoptimizer-1:

AdamOptimizer - 1 (com.microsoft)
=================================

**Version**

* **name**: `AdamOptimizer (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.AdamOptimizer>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

**Attributes**

* **alpha**:
  Coefficient of previous gradient in running average. Default value is ``?``.
* **beta**:
  Coefficient of previous squared gradient in running average.The
  effective learning rate is computed by r = R / (1 + T *
  decay_factor). Default to 0 so that increasing update counts doesn't
  reduce the learning rate. Default value is ``?``.
* **do_bias_correction**:
  Compute unbiased 1st and 2nd momentums. Default value is ``?``.
* **epsilon**:
  Small scalar to avoid dividing by zero. Default value is ``?``.
* **lambda**:
  Regularization coefficient of 0.5 * lambda * ||X||_2^2. Default to
  0, which means no regularization. Default value is ``?``.
* **max_norm_clip**:
  clip threshold of gradients. Default value is ``?``.
* **weight_decay_mode**:
  Modes for applying weight decay, 0 means applying decay before
  weight update, 1 means applying decay after weight update. Default value is ``?``.

**Inputs**

Between 6 and 10 inputs.

* **R** (heterogeneous) - **T1**:
  The initial learning rate.
* **T** (heterogeneous) - **T2**:
  The update count of "X". It should be a scalar.
* **weights** (heterogeneous) - **T3**:
  weights to optimize.
* **gradients** (heterogeneous) - **T_GRAD**:
  gradients computed in this iteration.
* **moment_1** (heterogeneous) - **T4**:
  exponentially averaged historical gradients.
* **moment_2** (heterogeneous) - **T4**:
  exponentially averaged historical squared gradients.
* **mixed_precision_weights** (optional, heterogeneous) - **T_MIXED_PRECISION_FP**:
  FP16 or BFloat16 weights to optimize.
* **loss_scale** (optional, heterogeneous) - **T3**:
  loss scale for mixed precision training
* **global_gradient_norm** (optional, heterogeneous) - **T_GRAD_NORM**:
  Global gradient norm.
* **update_signal** (optional, heterogeneous) - **T_BOOL**:
  This signal indicates if weight tensors should be updated.

**Outputs**

Between 3 and 6 outputs.

* **new_T** (heterogeneous) - **T2**:
  New update count.
* **new_moment_1** (heterogeneous) - **T4**:
  New averaged gradients.
* **new_moment_2** (heterogeneous) - **T4**:
  New averaged squared gradients.
* **new_weights** (optional, heterogeneous) - **T3**:
  New weights.
* **new_gradients** (optional, heterogeneous) - **T_GRAD**:
  New gradients.
* **new_mixed_precision_weights** (optional, heterogeneous) - **T_MIXED_PRECISION_FP**:
  New FP16 or BFloat16 weights

**Examples**
