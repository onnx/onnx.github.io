
.. _l-onnx-doc-NegativeLogLikelihoodLoss:

=========================
NegativeLogLikelihoodLoss
=========================

.. contents::
    :local:


.. _l-onnx-op-negativeloglikelihoodloss-13:

NegativeLogLikelihoodLoss - 13
==============================

**Version**

* **name**: `NegativeLogLikelihoodLoss (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#NegativeLogLikelihoodLoss>`_
* **domain**: **main**
* **since_version**: **13**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 13**.

**Summary**

A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.
Its "input" tensor has the shape of (N, C, d1, d2, ..., dk) where k >= 0.
The "input" tensor contains log-probabilities for input[n, :, d_1, d_2,..., d_k] being in a class of [0, C).
The operator's "target" input tensor has the shape of (N, d1, d2, ..., dk). It encodes class labels (one of C classes)
or it may contain a special value (indicated by an attribute ignore_index) for N x d1 x d2 x ... x dk samples.
The loss value for input[n, :, d_1, d_2,...d_k] being classified as class c = target[n][d_1][d_2]...[d_k] is computed as:

    loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k].

When an optional "weight" is provided, the sample loss is calculated as:

    loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k] * weight[c].

loss is zero for the case when target-value equals ignore_index.

    loss[n][d_1][d_2]...[d_k] = 0, when target[n][d_1][d_2]...[d_k] = ignore_index

If "reduction" attribute is set to "none", the operator's output will be the above loss with shape (N, d1, d2, ..., dk).
If "reduction" attribute is set to "mean" (the default attribute value), the output loss is (weight) averaged:

    mean(loss), if "weight" is not provided,

or if weight is provided,

    sum(loss) / sum(weight[target[n][d_1][d_2]...[d_k]]]), for all samples.

If "reduction" attribute is set to "sum", the output is a scalar:
    sum(loss).

See also https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss.

Example 1:

    // negative log likelihood loss, "none" reduction
    N, C, d1 = 2, 3, 2
    input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
             [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
    target = [[2, 1], [0, 2]]

    loss = np.zeros((N, d1))
    for n in range(N):
        for d_1 in range(d1):
            c = target[n][d_1]
            loss[n][d_1] = -input[n][c][d_1]

    // print(loss)
    // [[-3. -2.]
    //  [-0. -2.]]

Example 2:

    // weighted negative log likelihood loss, sum reduction
    N, C, d1 = 2, 3, 2
    input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
            [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
    target = [[2, 1], [0, 2]]
    weight = [0.2, 0.3, 0.1]
    loss = np.zeros((N, d1))
    for n in range(N):
        for d_1 in range(d1):
            c = target[n][d_1]
            loss[n][d_1] = -input[n][c][d_1] * weight[c]

    loss = np.sum(loss)
    // print(loss)
    // -1.1

Example 3:

    // weighted negative log likelihood loss, mean reduction
    N, C, d1 = 2, 3, 2
    input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
            [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
    target = [[2, 1], [0, 2]]
    weight = [0.2, 0.3, 0.1]
    loss = np.zeros((N, d1))
    weight_total = 0
    for n in range(N):
        for d_1 in range(d1):
            c = target[n][d_1]
            loss[n][d_1] = -input[n][c][d_1] * weight[c]
            weight_total = weight_total + weight[c]

    loss = np.sum(loss) / weight_total
    // print(loss)
    // -1.57

**Attributes**

* **ignore_index**:
  Specifies a target value that is ignored and does not contribute to
  the input gradient. It's an optional value.
* **reduction**:
  Type of reduction to apply to loss: none, sum, mean (default).
  'none': the output is the loss for each sample. 'sum': the output
  will be summed. 'mean': the sum of the output will be divided by the
  sum of applied weights. Default value is ``'mean'``.

**Inputs**

Between 2 and 3 inputs.

* **input** (heterogeneous) - **T**:
  Input tensor of shape (N, C) or (N, C, d1, d2, ..., dk).
* **target** (heterogeneous) - **Tind**:
  Target tensor of shape (N) or (N, d1, d2, ..., dk). Target element
  value shall be in range of [0, C). If ignore_index is specified, it
  may have a value outside [0, C) and the target values should either
  be in the range [0, C) or have the value ignore_index.
* **weight** (optional, heterogeneous) - **T**:
  Optional rescaling weight tensor. If given, it has to be a tensor of
  size C. Otherwise, it is treated as if having all ones.

**Outputs**

* **loss** (heterogeneous) - **T**:
  The negative log likelihood loss

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input, weight, and output types to floating-point tensors.
* **Tind** in (
  tensor(int32),
  tensor(int64)
  ):
  Constrain target to integer types

**Examples**

**input_shape_is_NC**

::

    reduction = 'none'
    node = onnx.helper.make_node(
        'NegativeLogLikelihoodLoss',
        inputs=['input', 'target'],
        outputs=['loss'],
        reduction=reduction
    )

    N, C = 3, 5
    np.random.seed(0)
    input = np.random.rand(N, C).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N, )).astype(np.int64)

    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=None, reduction=reduction)

    expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
        name='test_nllloss_NC')

**input_shape_is_NCd1d2**

::

    reduction = 'none'
    node = onnx.helper.make_node(
        'NegativeLogLikelihoodLoss',
        inputs=['input', 'target'],
        outputs=['loss'],
        reduction=reduction
    )

    N, C, dim1, dim2 = 3, 5, 6, 6
    np.random.seed(0)
    input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)

    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=None, reduction=reduction)

    expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
        name='test_nllloss_NCd1d2')

**input_shape_is_NCd1d2_reduction_mean**

::

    reduction = 'mean'
    node = onnx.helper.make_node(
        'NegativeLogLikelihoodLoss',
        inputs=['input', 'target'],
        outputs=['loss'],
        reduction=reduction
    )

    N, C, dim1, dim2 = 3, 5, 6, 6
    np.random.seed(0)
    input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)

    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=None, reduction=reduction)

    expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
        name='test_nllloss_NCd1d2_reduction_mean')

**input_shape_is_NCd1d2_reduction_sum**

::

    reduction = 'sum'
    node = onnx.helper.make_node(
        'NegativeLogLikelihoodLoss',
        inputs=['input', 'target'],
        outputs=['loss'],
        reduction=reduction
    )

    N, C, dim1, dim2 = 3, 5, 6, 6
    np.random.seed(0)
    input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)

    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=None, reduction=reduction)

    expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
        name='test_nllloss_NCd1d2_reduction_sum')

**input_shape_is_NCd1d2_with_weight**

::

    reduction = 'none'
    node = onnx.helper.make_node(
        'NegativeLogLikelihoodLoss',
        inputs=['input', 'target', 'weight'],
        outputs=['loss'],
        reduction=reduction
    )

    N, C, dim1, dim2 = 3, 5, 6, 6
    np.random.seed(0)
    input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)
    weight = np.random.rand(C).astype(np.float32)

    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=weight, reduction=reduction)

    expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss],
        name='test_nllloss_NCd1d2_with_weight')

**input_shape_is_NCd1d2_with_weight_reduction_mean**

::

    reduction = 'mean'
    node = onnx.helper.make_node(
        'NegativeLogLikelihoodLoss',
        inputs=['input', 'target', 'weight'],
        outputs=['loss'],
        reduction=reduction
    )

    N, C, dim1, dim2 = 3, 5, 6, 6
    np.random.seed(0)
    input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)
    weight = np.random.rand(C).astype(np.float32)

    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=weight, reduction=reduction)

    expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss],
        name='test_nllloss_NCd1d2_with_weight_reduction_mean')

**input_shape_is_NCd1d2_with_weight_reduction_sum**

::

    reduction = 'sum'
    node = onnx.helper.make_node(
        'NegativeLogLikelihoodLoss',
        inputs=['input', 'target', 'weight'],
        outputs=['loss'],
        reduction=reduction
    )

    N, C, dim1, dim2 = 3, 5, 6, 6
    np.random.seed(0)
    input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)
    weight = np.random.rand(C).astype(np.float32)

    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=weight, reduction=reduction)

    expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss],
        name='test_nllloss_NCd1d2_with_weight_reduction_sum')

**input_shape_is_NCd1d2_with_weight_reduction_sum_ii**

::

    reduction = 'sum'
    ignore_index = np.int64(0)
    node = onnx.helper.make_node(
        'NegativeLogLikelihoodLoss',
        inputs=['input', 'target', 'weight'],
        outputs=['loss'],
        reduction=reduction,
        ignore_index=ignore_index
    )

    N, C, dim1, dim2 = 3, 5, 6, 6
    np.random.seed(0)
    input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)
    target[0][0][0] = np.int64(0)
    weight = np.random.rand(C).astype(np.float32)

    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=weight, reduction=reduction, ignore_index=ignore_index)

    expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss],
        name='test_nllloss_NCd1d2_with_weight_reduction_sum_ii')

**input_shape_is_NCd1d2_no_weight_reduction_mean_ii**

::

    reduction = 'mean'
    ignore_index = np.int64(1)
    node = onnx.helper.make_node(
        'NegativeLogLikelihoodLoss',
        inputs=['input', 'target'],
        outputs=['loss'],
        reduction=reduction,
        ignore_index=ignore_index
    )

    N, C, dim1, dim2 = 3, 5, 6, 6
    np.random.seed(0)
    input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)
    target[0][0][0] = np.int64(1)

    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, reduction=reduction, ignore_index=ignore_index)

    expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
        name='test_nllloss_NCd1d2_no_weight_reduction_mean_ii')

**input_shape_is_NCd1**

::

    reduction = 'mean'
    node = onnx.helper.make_node(
        'NegativeLogLikelihoodLoss',
        inputs=['input', 'target'],
        outputs=['loss'],
        reduction=reduction
    )

    N, C, d1 = 3, 5, 2
    np.random.seed(0)
    input = np.random.rand(N, C, d1).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N, d1)).astype(np.int64)

    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=None, reduction=reduction)

    expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
        name='test_nllloss_NCd1')

**input_shape_is_NCd1_weight**

::

    reduction = 'mean'
    node = onnx.helper.make_node(
        'NegativeLogLikelihoodLoss',
        inputs=['input', 'target', 'weight'],
        outputs=['loss'],
        reduction=reduction
    )

    N, C, d1 = 3, 5, 2
    np.random.seed(0)
    input = np.random.rand(N, C, d1).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N, d1)).astype(np.int64)
    weight = np.random.rand(C).astype(np.float32)

    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=weight, reduction=reduction)

    expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss],
        name='test_nllloss_NCd1_weight')

**input_shape_is_NCd1_ii**

::

    reduction = 'mean'
    ignore_index = np.int64(1)
    node = onnx.helper.make_node(
        'NegativeLogLikelihoodLoss',
        inputs=['input', 'target'],
        outputs=['loss'],
        reduction=reduction,
        ignore_index=ignore_index
    )

    N, C, d1 = 3, 5, 2
    np.random.seed(0)
    input = np.random.rand(N, C, d1).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N, d1)).astype(np.int64)
    target[0][0] = np.int64(1)

    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=None, reduction=reduction, ignore_index=ignore_index)

    expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
        name='test_nllloss_NCd1_ii')

**input_shape_is_NCd1_weight_ii**

::

    reduction = 'mean'
    ignore_index = np.int64(1)
    node = onnx.helper.make_node(
        'NegativeLogLikelihoodLoss',
        inputs=['input', 'target', 'weight'],
        outputs=['loss'],
        reduction=reduction,
        ignore_index=ignore_index
    )

    N, C, d1 = 3, 5, 2
    np.random.seed(0)
    input = np.random.rand(N, C, d1).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N, d1)).astype(np.int64)
    target[0][0] = np.int64(1)
    weight = np.random.rand(C).astype(np.float32)

    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=weight, reduction=reduction, ignore_index=ignore_index)

    expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss],
        name='test_nllloss_NCd1_weight_ii')

**input_shape_is_NCd1d2d3d4d5_mean_weight**

::

    reduction = 'mean'

    node = onnx.helper.make_node(
        'NegativeLogLikelihoodLoss',
        inputs=['input', 'target', 'weight'],
        outputs=['loss'],
        reduction=reduction)

    N, C, dim1, dim2, dim3, dim4, dim5 = 3, 5, 6, 6, 5, 3, 4
    np.random.seed(0)
    input = np.random.rand(N, C, dim1, dim2, dim3, dim4, dim5).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3, dim4, dim5)).astype(np.int64)
    weight = np.random.rand(C).astype(np.float32)

    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input,
                                                                        target,
                                                                        weight=weight,
                                                                        reduction=reduction)

    expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss],
        name='test_nllloss_NCd1d2d3d4d5_mean_weight')

**input_shape_is_NCd1d2d3d4d5_none_no_weight**

::

    reduction = 'none'

    node = onnx.helper.make_node(
        'NegativeLogLikelihoodLoss',
        inputs=['input', 'target'],
        outputs=['loss'],
        reduction=reduction)

    N, C, dim1, dim2, dim3, dim4, dim5 = 3, 5, 6, 6, 5, 3, 4
    np.random.seed(0)
    input = np.random.rand(N, C, dim1, dim2, dim3, dim4, dim5).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3, dim4, dim5)).astype(np.int64)

    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input,
                                                                        target,
                                                                        reduction=reduction)

    expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
        name='test_nllloss_NCd1d2d3d4d5_none_no_weight')

**input_shape_is_NCd1_mean_weight_negative_ii**

::

    reduction = 'mean'
    ignore_index = np.int64(-1)

    node = onnx.helper.make_node(
        'NegativeLogLikelihoodLoss',
        inputs=['input', 'target', 'weight'],
        outputs=['loss'],
        reduction=reduction,
        ignore_index=ignore_index)

    N, C, dim1 = 3, 5, 6
    np.random.seed(0)
    input = np.random.rand(N, C, dim1).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N, dim1)).astype(np.int64)
    target[0][0] = -1
    weight = np.random.rand(C).astype(np.float32)

    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input,
                                                                        target,
                                                                        weight=weight,
                                                                        reduction=reduction,
                                                                        ignore_index=ignore_index)

    expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss],
        name='test_nllloss_NCd1_mean_weight_negative_ii')

**input_shape_is_NCd1d2d3_none_no_weight_negative_ii**

::

    reduction = 'none'
    ignore_index = np.int64(-5)

    node = onnx.helper.make_node(
        'NegativeLogLikelihoodLoss',
        inputs=['input', 'target'],
        outputs=['loss'],
        reduction=reduction,
        ignore_index=ignore_index)

    N, C, dim1, dim2, dim3 = 3, 5, 6, 6, 5
    np.random.seed(0)
    input = np.random.rand(N, C, dim1, dim2, dim3).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3)).astype(np.int64)
    target[0][0][0][0] = -5

    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input,
                                                                        target,
                                                                        reduction=reduction,
                                                                        ignore_index=ignore_index)

    expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
        name='test_nllloss_NCd1d2d3_none_no_weight_negative_ii')

**input_shape_is_NCd1d2d3_sum_weight_high_ii**

::

    reduction = 'sum'
    ignore_index = np.int64(10)

    node = onnx.helper.make_node(
        'NegativeLogLikelihoodLoss',
        inputs=['input', 'target', 'weight'],
        outputs=['loss'],
        reduction=reduction,
        ignore_index=ignore_index)

    N, C = 3, 5
    np.random.seed(0)
    input = np.random.rand(N, C).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N)).astype(np.int64)
    target[0] = 10
    weight = np.random.rand(C).astype(np.float32)

    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input,
                                                                        target,
                                                                        weight=weight,
                                                                        reduction=reduction,
                                                                        ignore_index=ignore_index)

    expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss],
        name='test_nllloss_NCd1d2d3_sum_weight_high_ii')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Its "input" tensor has the shape of (N, C, d1, d2, ..., dk) where k >= 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Its "input" tensor has the shape of (N, C, d1, d2, ..., dk) where k >= 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The "input" tensor contains log-probabilities for input[n, :, d_1, d_2,..., d_k] being in a class of [0, C).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The "input" tensor contains log-probabilities for input[n, :, d_1, d_2,..., d_k] being in a class of [0, C).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The operator's "target" input tensor has the shape of (N, d1, d2, ..., dk). It encodes class labels (one of C classes)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The operator's "target" input tensor has the shape of (N, d1, d2, ..., dk). It encodes class labels (one of C classes)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">or it may contain a special value (indicated by an attribute ignore_index) for N x d1 x d2 x ... x dk samples.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">or it may contain a special value (indicated by an attribute ignore_index) for N x d1 x d2 x ... x dk samples.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The loss value for input[n, :, d_1, d_2,...d_k] being classified as class c = target[n][d_1][d_2]...[d_k] is computed as:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The loss value for input[n, :, d_1, d_2,...d_k] being classified as class c = target[n][d_1][d_2]...[d_k] is computed as:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">6</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k].</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">8</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">When an optional "weight" is provided, the sample loss is calculated as:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">When an optional "weight" is provided, the sample loss is calculated as:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">10</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k] * weight[c].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k] * weight[c].</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">12</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">loss is zero for the case when target-value equals ignore_index.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">loss is zero for the case when target-value equals ignore_index.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    loss[n][d_1][d_2]...[d_k] = 0, when target[n][d_1][d_2]...[d_k] = ignore_index</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    loss[n][d_1][d_2]...[d_k] = 0, when target[n][d_1][d_2]...[d_k] = ignore_index</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">16</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If "reduction" attribute is set to "none", the operator's output will be the above loss with shape (N, d1, d2, ..., dk).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If "reduction" attribute is set to "none", the operator's output will be the above loss with shape (N, d1, d2, ..., dk).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If "reduction" attribute is set to "mean" (the default attribute value), the output loss is (weight) averaged:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If "reduction" attribute is set to "mean" (the default attribute value), the output loss is (weight) averaged:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">19</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    mean(loss), if "weight" is not provided,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    mean(loss), if "weight" is not provided,</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">21</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">or if weight is provided,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">or if weight is provided,</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">23</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    sum(loss) / sum(weight[target[n][d_1][d_2]...[d_k]]]), for all samples.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    sum(loss) / sum(weight[target[n][d_1][d_2]...[d_k]]]), for all samples.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">25</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If "reduction" attribute is set to "sum", the output is a scalar:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If "reduction" attribute is set to "sum", the output is a scalar:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    sum(loss).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    sum(loss).</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">28</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">See also https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">See also https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">30</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 1:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 1:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">32</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    // negative log likelihood loss, "none" reduction</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    // negative log likelihood loss, "none" reduction</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    N, C, d1 = 2, 3, 2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    N, C, d1 = 2, 3, 2</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">             [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">             [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    target = [[2, 1], [0, 2]]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    target = [[2, 1], [0, 2]]</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">38</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    loss = np.zeros((N, d1))</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    loss = np.zeros((N, d1))</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    for n in range(N):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    for n in range(N):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for d_1 in range(d1):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for d_1 in range(d1):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">            c = target[n][d_1]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">            c = target[n][d_1]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">            loss[n][d_1] = -input[n][c][d_1]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">            loss[n][d_1] = -input[n][c][d_1]</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">44</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    // print(loss)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    // print(loss)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    // [[-3. -2.]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    // [[-3. -2.]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    //  [-0. -2.]]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    //  [-0. -2.]]</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">48</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 2:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 2:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">50</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    // weighted negative log likelihood loss, sum reduction</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    // weighted negative log likelihood loss, sum reduction</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    N, C, d1 = 2, 3, 2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    N, C, d1 = 2, 3, 2</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">            [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">            [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    target = [[2, 1], [0, 2]]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    target = [[2, 1], [0, 2]]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    weight = [0.2, 0.3, 0.1]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    weight = [0.2, 0.3, 0.1]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    loss = np.zeros((N, d1))</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    loss = np.zeros((N, d1))</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    for n in range(N):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    for n in range(N):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for d_1 in range(d1):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for d_1 in range(d1):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">            c = target[n][d_1]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">            c = target[n][d_1]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">            loss[n][d_1] = -input[n][c][d_1] * weight[c]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">            loss[n][d_1] = -input[n][c][d_1] * weight[c]</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">62</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    loss = np.sum(loss)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    loss = np.sum(loss)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    // print(loss)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    // print(loss)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    // -1.1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    // -1.1</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">66</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 3:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Example 3:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">68</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    // weighted negative log likelihood loss, mean reduction</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    // weighted negative log likelihood loss, mean reduction</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    N, C, d1 = 2, 3, 2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    N, C, d1 = 2, 3, 2</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">            [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">            [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    target = [[2, 1], [0, 2]]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    target = [[2, 1], [0, 2]]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    weight = [0.2, 0.3, 0.1]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    weight = [0.2, 0.3, 0.1]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    loss = np.zeros((N, d1))</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    loss = np.zeros((N, d1))</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    weight_total = 0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    weight_total = 0</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    for n in range(N):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    for n in range(N):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for d_1 in range(d1):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for d_1 in range(d1):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">            c = target[n][d_1]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">            c = target[n][d_1]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">            loss[n][d_1] = -input[n][c][d_1] * weight[c]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">            loss[n][d_1] = -input[n][c][d_1] * weight[c]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">            weight_total = weight_total + weight[c]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">            weight_total = weight_total + weight[c]</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">82</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    loss = np.sum(loss) / weight_total</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    loss = np.sum(loss) / weight_total</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    // print(loss)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    // print(loss)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    // -1.57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    // -1.57</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **ignore_index**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **ignore_index**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Specifies a target value that is ignored and does not contribute to</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Specifies a target value that is ignored and does not contribute to</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the input gradient. It's an optional value.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the input gradient. It's an optional value.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **reduction**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **reduction**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Type of reduction to apply to loss: none, sum, mean (default).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Type of reduction to apply to loss: none, sum, mean (default).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'none': the output is the loss for each sample. 'sum': the output</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'none': the output is the loss for each sample. 'sum': the output</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  will be summed. 'mean': the sum of the output will be divided by the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  will be summed. 'mean': the sum of the output will be divided by the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  sum of applied weights. Default value is 'mean'.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  sum of applied weights. Default value is 'mean'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 3 inputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 3 inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **input** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **input** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor of shape (N, C) or (N, C, d1, d2, ..., dk).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input tensor of shape (N, C) or (N, C, d1, d2, ..., dk).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **target** (heterogeneous) - **Tind**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **target** (heterogeneous) - **Tind**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Target tensor of shape (N) or (N, d1, d2, ..., dk). Target element</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Target tensor of shape (N) or (N, d1, d2, ..., dk). Target element</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  value shall be in range of [0, C). If ignore_index is specified, it</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  value shall be in range of [0, C). If ignore_index is specified, it</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  may have a value outside [0, C) and the target values should either</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  may have a value outside [0, C) and the target values should either</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  be in the range [0, C) or have the value ignore_index.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  be in the range [0, C) or have the value ignore_index.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">109</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **weight** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **weight** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">110</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional rescaling weight tensor. If given, it has to be a tensor of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Optional rescaling weight tensor. If given, it has to be a tensor of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">111</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  size C. Otherwise, it is treated as if having all ones.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  size C. Otherwise, it is treated as if having all ones.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">112</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">113</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">114</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">115</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **loss** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **loss** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">116</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The negative log likelihood loss</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The negative log likelihood loss</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">117</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">118</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">119</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">120</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">121</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">122</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">123</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">124</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">125</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input, weight, and output types to floating-point tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input, weight, and output types to floating-point tensors.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">126</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Tind** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Tind** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">127</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">128</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">109</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">129</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">110</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">130</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain target to integer types</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain target to integer types</code></td></tr>
    </table>

.. _l-onnx-op-negativeloglikelihoodloss-12:

NegativeLogLikelihoodLoss - 12
==============================

**Version**

* **name**: `NegativeLogLikelihoodLoss (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#NegativeLogLikelihoodLoss>`_
* **domain**: **main**
* **since_version**: **12**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 12**.

**Summary**

A NegativeLogLikelihoodLoss operator computes (weighted) negative log likelihood loss.
Its "input" tensor has the shape of (N, C, d1, d2, ..., dk) where k >= 0.
The "input" tensor contains log-probabilities for input[n, :, d_1, d_2,..., d_k] being in a class of [0, C).
The operator's "target" input tensor has the shape of (N, d1, d2, ..., dk). It encodes class labels (one of C classes)
or it may contain a special value (indicated by an attribute ignore_index) for N x d1 x d2 x ... x dk samples.
The loss value for input[n, :, d_1, d_2,...d_k] being classified as class c = target[n][d_1][d_2]...[d_k] is computed as:
    loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k].
When an optional "weight" is provided, the sample loss is calculated as:
    loss[n][d_1][d_2]...[d_k] = -input[n][c][d_1][d_2]...[d_k] * weight[c].
loss is zero for the case when target-value equals ignore_index.

    loss[n][d_1][d_2]...[d_k] = 0, when target[n][d_1][d_2]...[d_k] = ignore_index
If "reduction" attribute is set to "none", the operator's output will be the above loss with shape (N, d1, d2, ..., dk).
If "reduction" attribute is set to "mean" (the default attribute value), the output loss is (weight) averaged:
    mean(loss), if "weight" is not provided,
or if weight is provided,
    sum(loss) / sum(weight[target[n][d_1][d_2]...[d_k]]]), for all samples.
If "reduction" attribute is set to "sum", the output is a scalar:
    sum(loss).
See also https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss.
Example 1:
    // negative log likelihood loss, "none" reduction
    N, C, d1 = 2, 3, 2
    input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
             [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
    target = [[2, 1], [0, 2]]
    loss = np.zeros((N, d1))
    for n in range(N):
        for d_1 in range(d1):
            c = target[n][d_1]
            loss[n][d_1] = -input[n][c][d_1]
    // print(loss)
    // [[-3. -2.]
    //  [-0. -2.]]
Example 2:
    // weighted negative log likelihood loss, sum reduction
    N, C, d1 = 2, 3, 2
    input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
            [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
    target = [[2, 1], [0, 2]]
    weight = [0.2, 0.3, 0.1]
    loss = np.zeros((N, d1))
    for n in range(N):
        for d_1 in range(d1):
            c = target[n][d_1]
            loss[n][d_1] = -input[n][c][d_1] * weight[c]
    loss = np.sum(loss)
    // print(loss)
    // -1.1
Example 3:
    // weighted negative log likelihood loss, mean reduction
    N, C, d1 = 2, 3, 2
    input = [[[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
            [[0.0, 1.0], [2.0, 2.0], [1.0, 2]]]
    target = [[2, 1], [0, 2]]
    weight = [0.2, 0.3, 0.1]
    loss = np.zeros((N, d1))
    weight_total = 0
    for n in range(N):
        for d_1 in range(d1):
            c = target[n][d_1]
            loss[n][d_1] = -input[n][c][d_1] * weight[c]
            weight_total = weight_total + weight[c]
    loss = np.sum(loss) / weight_total
    // print(loss)
    // -1.57

**Attributes**

* **ignore_index**:
  Specifies a target value that is ignored and does not contribute to
  the input gradient. It's an optional value.
* **reduction**:
  Type of reduction to apply to loss: none, sum, mean (default).
  'none': the output is the loss for each sample. 'sum': the output
  will be summed. 'mean': the sum of the output will be divided by the
  sum of applied weights. Default value is ``'mean'``.

**Inputs**

Between 2 and 3 inputs.

* **input** (heterogeneous) - **T**:
  Input tensor of shape (N, C) or (N, C, d1, d2, ..., dk).
* **target** (heterogeneous) - **Tind**:
  Target tensor of shape (N) or (N, d1, d2, ..., dk). Target element
  value shall be in range of [0, C). If ignore_index is specified, it
  may have a value outside [0, C) and the target values should either
  be in the range [0, C) or have the value ignore_index.
* **weight** (optional, heterogeneous) - **T**:
  Optional rescaling weight tensor. If given, it has to be a tensor of
  size C. Otherwise, it is treated as if having all ones.

**Outputs**

* **loss** (heterogeneous) - **T**:
  The negative log likelihood loss

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input, weight, and output types to floating-point tensors.
* **Tind** in (
  tensor(int32),
  tensor(int64)
  ):
  Constrain target to integer types
