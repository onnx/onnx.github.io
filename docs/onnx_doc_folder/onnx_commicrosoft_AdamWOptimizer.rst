
.. _l-onnx-doccom.microsoft-AdamWOptimizer:

==============================
com.microsoft - AdamWOptimizer
==============================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-adamwoptimizer-1:

AdamWOptimizer - 1 (com.microsoft)
==================================

**Version**

* **name**: `AdamWOptimizer (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.AdamWOptimizer>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

**Attributes**

* **adam_mode**:
  Modes for applying bias correction and weight decay (default 0) 0 :
  Weight decay is applied before weight is updated.  Computation
  aligned with Torch AdamW. In this mode,   correct_bias should be 1
  to keep aligned with PyTorch.1 : Weight decay is applied after
  weight is updated.    Computation is aligned with Huggingface AdamW. Default value is ``?``.
* **alpha**:
  Coefficient of previously accumulated gradient in running average. Default value is ``?``.
* **beta**:
  Coefficient of previously accumulated squared-gradient in running
  average. Default value is ``?``.
* **correct_bias**:
  Whether or not to correct bias, enabled by default. Default value is ``?``.
* **epsilon**:
  Small scalar to avoid dividing by zero. Default value is ``?``.
* **weight_decay**:
  weight decay coefficient. Default value is ``?``.

**Inputs**

Between 6 and 7 inputs.

* **lr** (heterogeneous) - **T1**:
  The learning rate.
* **step** (heterogeneous) - **T2**:
  The update count of weights. It should be a scalar.
* **weights** (heterogeneous) - **S_WEIGHT**:
  Sequence of weights to optimize.
* **gradients** (heterogeneous) - **S_GRAD**:
  Sequence of gradients computed in this iteration.
* **momentums_1** (heterogeneous) - **S_MOMENT**:
  Sequence of exponentially averaged historical gradients.
* **momentums_2** (heterogeneous) - **S_MOMENT**:
  Sequence of exponentially averaged historical squared gradients.
* **update_signal** (optional, heterogeneous) - **T_BOOL**:
  This signal indicates if weight updates are skipped, applicable to
  gradient infinity check in mixed precision training.

**Outputs**

Between 1 and 4 outputs.

* **updated_flag** (heterogeneous) - **T2**:
  Whether gradient is applied or not.
* **updated_weights** (optional, heterogeneous) - **S_WEIGHT**:
  Sequence of weights after optimize.
* **updated_momentums_1** (optional, heterogeneous) - **S_MOMENT**:
  Sequence of momentum_1 after optimize.
* **updated_momentums_2** (optional, heterogeneous) - **S_MOMENT**:
  Sequence of momentum_2 after optimize.

**Examples**
