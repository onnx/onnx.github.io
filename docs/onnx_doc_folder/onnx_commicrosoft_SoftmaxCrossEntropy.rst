
.. _l-onnx-doccom.microsoft-SoftmaxCrossEntropy:

===================================
com.microsoft - SoftmaxCrossEntropy
===================================

.. contents::
    :local:


.. _l-onnx-opcom-microsoft-softmaxcrossentropy-1:

SoftmaxCrossEntropy - 1 (com.microsoft)
=======================================

**Version**

* **name**: `SoftmaxCrossEntropy (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#com.microsoft.SoftmaxCrossEntropy>`_
* **domain**: **com.microsoft**
* **since_version**: **1**
* **function**:
* **support_level**:
* **shape inference**:

This version of the operator has been available
**since version 1 of domain com.microsoft**.

**Summary**

SoftmaxCrossEntropy

**Attributes**

* **reduction**:
  Type of reduction to apply to loss: none, sum, mean(default).
  'none': the output is the loss for each sample in the batch.'sum':
  the output will be summed. 'mean': the sum of the output will be
  divided by the batch_size. Default value is ``?``.

**Inputs**

* **logits** (heterogeneous) - **T**:
  Unscaled log probabilities, N-D input of shape (-1, num_classes).
* **label** (heterogeneous) - **T**:
  The onehot label is N-D input with the same shape as logits.

**Outputs**

Between 1 and 2 outputs.

* **Y** (heterogeneous) - **T**:
  loss.
* **log_prob** (optional, heterogeneous) - **T**:
  logsoftmax(logits)

**Examples**

**softmaxcrossentropy_none**

::

    # Define operator attributes.
    reduction = 'none'

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y'],
                                 outputs=['z'],
                                 reduction=reduction)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)

    # Compute SoftmaxCrossEntropyLoss
    sce = softmaxcrossentropy(x, labels, reduction='none')

    # Check results
    expect(node, inputs=[x, labels], outputs=[sce], name='test_sce_none')

**softmaxcrossentropy_none_log_prob**

::

    # Define operator attributes.
    reduction = 'none'

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y'],
                                 outputs=['z', 'log_prob'],
                                 reduction=reduction)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)

    # Compute SoftmaxCrossEntropyLoss
    loss, log_prob = softmaxcrossentropy(x, labels, reduction='none', get_log_prob=True)

    # Check results
    expect(node, inputs=[x, labels], outputs=[loss, log_prob], name='test_sce_none_log_prob')

**softmaxcrossentropy_none_weights**

::

    # Define operator attributes.
    reduction = 'none'

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y', 'w'],
                                 outputs=['z'],
                                 reduction=reduction)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)
    weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)

    # Compute SoftmaxCrossEntropyLoss
    sce = softmaxcrossentropy(x, labels, weight=weights, reduction='none')

    # Check results
    expect(node, inputs=[x, labels, weights], outputs=[sce], name='test_sce_none_weights')

**softmaxcrossentropy_none_weights_log_prob**

::

    # Define operator attributes.
    reduction = 'none'

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y', 'w'],
                                 outputs=['z', 'log_prob'],
                                 reduction=reduction)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)
    weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)

    # Compute SoftmaxCrossEntropyLoss
    loss, log_prob = softmaxcrossentropy(x, labels, weight=weights, reduction='none', get_log_prob=True)

    # Check results
    expect(node, inputs=[x, labels, weights], outputs=[loss, log_prob], name='test_sce_none_weights_log_prob')

**softmaxcrossentropy_sum**

::

    # Define operator attributes.
    reduction = 'sum'

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y'],
                                 outputs=['z'],
                                 reduction=reduction)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)

    # Compute SoftmaxCrossEntropyLoss
    sce = softmaxcrossentropy(x, labels, reduction='sum')

    # Check results
    expect(node, inputs=[x, labels], outputs=[sce], name='test_sce_sum')

**softmaxcrossentropy_sum_log_prob**

::

    # Define operator attributes.
    reduction = 'sum'

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y'],
                                 outputs=['z', 'log_prob'],
                                 reduction=reduction)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)

    # Compute SoftmaxCrossEntropyLoss
    loss, log_prob = softmaxcrossentropy(x, labels, reduction='sum', get_log_prob=True)

    # Check results
    expect(node, inputs=[x, labels], outputs=[loss, log_prob], name='test_sce_sum_log_prob')

**softmaxcrossentropy_mean**

::

    # Define operator attributes.
    reduction = 'mean'

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y'],
                                 outputs=['z'],
                                 reduction=reduction)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)

    # Compute SoftmaxCrossEntropyLoss
    sce = softmaxcrossentropy(x, labels)

    # Check results
    expect(node, inputs=[x, labels], outputs=[sce], name='test_sce_mean')

**softmaxcrossentropy_mean_log_prob**

::

    # Define operator attributes.
    reduction = 'mean'

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y'],
                                 outputs=['z', 'log_prob'],
                                 reduction=reduction)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)

    # Compute SoftmaxCrossEntropyLoss
    loss, log_prob = softmaxcrossentropy(x, labels, get_log_prob=True)

    # Check results
    expect(node, inputs=[x, labels], outputs=[loss, log_prob], name='test_sce_mean_log_prob')

**softmaxcrossentropy_mean_3d**

::

    # Define operator attributes.
    reduction = 'mean'

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y'],
                                 outputs=['z'],
                                 reduction=reduction)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5, 2).astype(np.float32)
    y = np.random.randint(0, high=5, size=(3, 2)).astype(np.int64)

    # Compute SoftmaxCrossEntropyLoss
    sce = softmaxcrossentropy(x, y)

    # Check results
    expect(node, inputs=[x, y], outputs=[sce], name='test_sce_mean_3d')

**softmaxcrossentropy_mean_3d_log_prob**

::

    # Define operator attributes.
    reduction = 'mean'

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y'],
                                 outputs=['z', 'log_prob'],
                                 reduction=reduction)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5, 2).astype(np.float32)
    y = np.random.randint(0, high=5, size=(3, 2)).astype(np.int64)

    # Compute SoftmaxCrossEntropyLoss
    loss, log_prob = softmaxcrossentropy(x, y, get_log_prob=True)

    # Check results
    expect(node, inputs=[x, y], outputs=[loss, log_prob], name='test_sce_mean_3d_log_prob')

**softmaxcrossentropy_mean_weights**

::

    # Define operator attributes.
    reduction = 'mean'

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y', 'w'],
                                 outputs=['z'],
                                 reduction=reduction)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)
    weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)

    # Compute SoftmaxCrossEntropyLoss
    sce = softmaxcrossentropy(x, labels, weight=weights)

    # Check results
    expect(node, inputs=[x, labels, weights], outputs=[sce], name='test_sce_mean_weight')

**softmaxcrossentropy_mean_weights_log_prob**

::

    # Define operator attributes.
    reduction = 'mean'

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y', 'w'],
                                 outputs=['z', 'log_prob'],
                                 reduction=reduction)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)
    weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)

    # Compute SoftmaxCrossEntropyLoss
    loss, log_prob = softmaxcrossentropy(x, labels, weight=weights, get_log_prob=True)

    # Check results
    expect(node, inputs=[x, labels, weights], outputs=[loss, log_prob], name='test_sce_mean_weight_log_prob')

**softmaxcrossentropy_mean_weights_ii**

::

    # Define operator attributes.
    reduction = 'mean'
    ignore_index = np.int64(0)

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y', 'w'],
                                 outputs=['z'],
                                 reduction=reduction,
                                 ignore_index=ignore_index)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)
    labels[0] = np.int64(0)
    weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)

    # Compute SoftmaxCrossEntropyLoss
    sce = softmaxcrossentropy(x, labels, weight=weights, ignore_index=ignore_index)

    # Check results
    expect(node, inputs=[x, labels, weights], outputs=[sce], name='test_sce_mean_weight_ii')

**softmaxcrossentropy_mean_weights_ii_log_prob**

::

    # Define operator attributes.
    reduction = 'mean'
    ignore_index = np.int64(0)

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y', 'w'],
                                 outputs=['z', 'log_prob'],
                                 reduction=reduction,
                                 ignore_index=ignore_index)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)
    labels[0] = np.int64(0)
    weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)

    # Compute SoftmaxCrossEntropyLoss
    loss, log_prob = softmaxcrossentropy(x, labels, weight=weights, ignore_index=ignore_index, get_log_prob=True)

    # Check results
    expect(node, inputs=[x, labels, weights], outputs=[loss, log_prob], name='test_sce_mean_weight_ii_log_prob')

**softmaxcrossentropy_mean_no_weights_ii**

::

    # Define operator attributes.
    reduction = 'mean'
    ignore_index = np.int64(2)

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                inputs=['x', 'y'],
                                outputs=['z'],
                                reduction=reduction,
                                ignore_index=ignore_index)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)
    labels[0] = np.int64(2)

    # Compute SoftmaxCrossEntropyLoss
    sce = softmaxcrossentropy(x, labels, ignore_index=ignore_index)

    # Check results
    expect(node, inputs=[x, labels], outputs=[sce], name='test_sce_mean_no_weight_ii')

**softmaxcrossentropy_mean_no_weights_ii_log_prob**

::

    # Define operator attributes.
    reduction = 'mean'
    ignore_index = np.int64(2)

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                inputs=['x', 'y'],
                                outputs=['z', 'log_prob'],
                                reduction=reduction,
                                ignore_index=ignore_index)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)
    labels[0] = np.int64(2)

    # Compute SoftmaxCrossEntropyLoss
    loss, log_prob = softmaxcrossentropy(x, labels, ignore_index=ignore_index, get_log_prob=True)

    # Check results
    expect(node, inputs=[x, labels], outputs=[loss, log_prob], name='test_sce_mean_no_weight_ii_log_prob')

**softmaxcrossentropy_mean_weights_ii_3d**

::

    # Define operator attributes.
    reduction = 'mean'
    ignore_index = np.int64(1)

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                inputs=['x', 'y', 'w'],
                                outputs=['z'],
                                reduction=reduction,
                                ignore_index=ignore_index)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5, 2).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, 2)).astype(np.int64)
    labels[0][0] = np.int64(1)
    weights = np.array([0.2, 0.3, 0.6, 0.1, 0.5], dtype=np.float32)

    # Compute SoftmaxCrossEntropyLoss
    sce = softmaxcrossentropy(x, labels, weight=weights, ignore_index=ignore_index)

    # Check results
    expect(node, inputs=[x, labels, weights], outputs=[sce], name='test_sce_mean_weight_ii_3d')

**softmaxcrossentropy_mean_weights_ii_3d_log_prob**

::

    # Define operator attributes.
    reduction = 'mean'
    ignore_index = np.int64(1)

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                inputs=['x', 'y', 'w'],
                                outputs=['z', 'log_prob'],
                                reduction=reduction,
                                ignore_index=ignore_index)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5, 2).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, 2)).astype(np.int64)
    labels[0][0] = np.int64(1)
    weights = np.array([0.2, 0.3, 0.6, 0.1, 0.5], dtype=np.float32)

    # Compute SoftmaxCrossEntropyLoss
    loss, log_prob = softmaxcrossentropy(x, labels, weight=weights, ignore_index=ignore_index, get_log_prob=True)

    # Check results
    expect(node, inputs=[x, labels, weights], outputs=[loss, log_prob], name='test_sce_mean_weight_ii_3d_log_prob')

**softmaxcrossentropy_mean_no_weights_ii_3d**

::

    # Define operator attributes.
    reduction = 'mean'
    ignore_index = np.int64(2)

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                inputs=['x', 'y'],
                                outputs=['z'],
                                reduction=reduction,
                                ignore_index=ignore_index)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5, 2).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, 2)).astype(np.int64)
    labels[0][0] = np.int64(2)

    # Compute SoftmaxCrossEntropyLoss
    sce = softmaxcrossentropy(x, labels, ignore_index=ignore_index)

    # Check results
    expect(node, inputs=[x, labels], outputs=[sce], name='test_sce_mean_no_weight_ii_3d')

**softmaxcrossentropy_mean_no_weights_ii_3d_log_prob**

::

    # Define operator attributes.
    reduction = 'mean'
    ignore_index = np.int64(2)

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                inputs=['x', 'y'],
                                outputs=['z', 'log_prob'],
                                reduction=reduction,
                                ignore_index=ignore_index)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5, 2).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, 2)).astype(np.int64)
    labels[0][0] = np.int64(2)

    # Compute SoftmaxCrossEntropyLoss
    loss, log_prob = softmaxcrossentropy(x, labels, ignore_index=ignore_index, get_log_prob=True)

    # Check results
    expect(node, inputs=[x, labels], outputs=[loss, log_prob], name='test_sce_mean_no_weight_ii_3d_log_prob')

**softmaxcrossentropy_mean_weights_ii_4d**

::

    # Define operator attributes.
    reduction = 'mean'
    ignore_index = np.int64(2)

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                inputs=['x', 'y', 'w'],
                                outputs=['z'],
                                reduction=reduction,
                                ignore_index=ignore_index)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5, 2, 7).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, 2, 7)).astype(np.int64)
    labels[0][0][0] = np.int64(2)
    weights = np.array([0.2, 0.3, 0.6, 0.1, 0.5], dtype=np.float32)

    # Compute SoftmaxCrossEntropyLoss
    sce = softmaxcrossentropy(x, labels, reduction=reduction, weight=weights, ignore_index=ignore_index)

    # Check results
    expect(node, inputs=[x, labels, weights], outputs=[sce], name='test_sce_mean_weight_ii_4d')

**softmaxcrossentropy_mean_weights_ii_4d_log_prob**

::

    # Define operator attributes.
    reduction = 'mean'
    ignore_index = np.int64(2)

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                inputs=['x', 'y', 'w'],
                                outputs=['z', 'log_prob'],
                                reduction=reduction,
                                ignore_index=ignore_index)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5, 2, 7).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, 2, 7)).astype(np.int64)
    labels[0][0][0] = np.int64(2)
    weights = np.array([0.2, 0.3, 0.6, 0.1, 0.5], dtype=np.float32)

    # Compute SoftmaxCrossEntropyLoss
    loss, log_prob = softmaxcrossentropy(x, labels, reduction=reduction, weight=weights, ignore_index=ignore_index, get_log_prob=True)

    # Check results
    expect(node, inputs=[x, labels, weights], outputs=[loss, log_prob], name='test_sce_mean_weight_ii_4d_log_prob')

**softmaxcrossentropy_mean_no_weights_ii_4d**

::

    # Define operator attributes.
    reduction = 'mean'
    ignore_index = np.int64(2)

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                inputs=['x', 'y'],
                                outputs=['z'],
                                reduction=reduction,
                                ignore_index=ignore_index)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5, 2, 7).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, 2, 7)).astype(np.int64)
    labels[0][0][0] = np.int64(2)

    # Compute SoftmaxCrossEntropyLoss
    sce = softmaxcrossentropy(x, labels, reduction=reduction, ignore_index=ignore_index)

    # Check results
    expect(node, inputs=[x, labels], outputs=[sce], name='test_sce_mean_no_weight_ii_4d')

**softmaxcrossentropy_mean_no_weights_ii_4d_log_prob**

::

    # Define operator attributes.
    reduction = 'mean'
    ignore_index = np.int64(2)

    # Create operator.
    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                inputs=['x', 'y'],
                                outputs=['z', 'log_prob'],
                                reduction=reduction,
                                ignore_index=ignore_index)

    # Define operator inputs.
    np.random.seed(0)
    x = np.random.rand(3, 5, 2, 7).astype(np.float32)
    labels = np.random.randint(0, high=5, size=(3, 2, 7)).astype(np.int64)
    labels[0][0][0] = np.int64(2)

    # Compute SoftmaxCrossEntropyLoss
    loss, log_prob = softmaxcrossentropy(x, labels, reduction=reduction, ignore_index=ignore_index, get_log_prob=True)

    # Check results
    expect(node, inputs=[x, labels], outputs=[loss, log_prob], name='test_sce_mean_no_weight_ii_4d_log_prob')

**input_shape_is_NCd1d2d3d4d5_mean_weight**

::

    reduction = 'mean'

    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y', 'w'],
                                 outputs=['z'],
                                 reduction=reduction)

    N, C, dim1, dim2, dim3, dim4, dim5 = 3, 5, 6, 6, 5, 3, 4
    np.random.seed(0)
    x = np.random.rand(N, C, dim1, dim2, dim3, dim4, dim5).astype(np.float32)
    labels = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3, dim4, dim5)).astype(np.int64)
    weight = np.random.rand(C).astype(np.float32)

    sce = softmaxcrossentropy(x,
                            labels,
                            weight=weight,
                            reduction=reduction)

    expect(node, inputs=[x, labels, weight], outputs=[sce], name='test_sce_NCd1d2d3d4d5_mean_weight')

**input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob**

::

    reduction = 'mean'

    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y', 'w'],
                                 outputs=['z', 'log_prob'],
                                 reduction=reduction)

    N, C, dim1, dim2, dim3, dim4, dim5 = 3, 5, 6, 6, 5, 3, 4
    np.random.seed(0)
    x = np.random.rand(N, C, dim1, dim2, dim3, dim4, dim5).astype(np.float32)
    labels = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3, dim4, dim5)).astype(np.int64)
    weight = np.random.rand(C).astype(np.float32)

    loss, log_prob = softmaxcrossentropy(x,
                            labels,
                            weight=weight,
                            reduction=reduction,
                            get_log_prob=True)

    expect(node, inputs=[x, labels, weight], outputs=[loss, log_prob], name='test_sce_NCd1d2d3d4d5_mean_weight_log_prob')

**input_shape_is_NCd1d2d3d4d5_none_no_weight**

::

    reduction = 'none'

    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y'],
                                 outputs=['z'],
                                 reduction=reduction)

    N, C, dim1, dim2, dim3, dim4, dim5 = 3, 5, 6, 6, 5, 3, 4
    np.random.seed(0)
    x = np.random.rand(N, C, dim1, dim2, dim3, dim4, dim5).astype(np.float32)
    labels = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3, dim4, dim5)).astype(np.int64)

    sce = softmaxcrossentropy(x,
                            labels,
                            reduction=reduction)

    expect(node, inputs=[x, labels], outputs=[sce], name='test_sce_NCd1d2d3d4d5_none_no_weight')

**input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob**

::

    reduction = 'none'

    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y'],
                                 outputs=['z', 'log_prob'],
                                 reduction=reduction)

    N, C, dim1, dim2, dim3, dim4, dim5 = 3, 5, 6, 6, 5, 3, 4
    np.random.seed(0)
    x = np.random.rand(N, C, dim1, dim2, dim3, dim4, dim5).astype(np.float32)
    labels = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3, dim4, dim5)).astype(np.int64)

    loss, log_prob = softmaxcrossentropy(x,
                            labels,
                            reduction=reduction,
                            get_log_prob=True)

    expect(node, inputs=[x, labels], outputs=[loss, log_prob], name='test_sce_NCd1d2d3d4d5_none_no_weight_log_prob')

**input_shape_is_NCd1_mean_weight_negative_ii**

::

    reduction = 'mean'
    ignore_index = np.int64(-1)

    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y', 'w'],
                                 outputs=['z'],
                                 reduction=reduction,
                                 ignore_index=ignore_index)

    N, C, dim1 = 3, 5, 6
    np.random.seed(0)
    x = np.random.rand(N, C, dim1).astype(np.float32)
    labels = np.random.randint(0, high=C, size=(N, dim1)).astype(np.int64)
    labels[0][0] = -1
    weight = np.random.rand(C).astype(np.float32)

    sce = softmaxcrossentropy(x,
                              labels,
                              weight=weight,
                              reduction=reduction,
                              ignore_index=ignore_index)

    expect(node, inputs=[x, labels, weight], outputs=[sce], name='test_sce_NCd1_mean_weight_negative_ii')

**input_shape_is_NCd1_mean_weight_negative_ii_log_prob**

::

    reduction = 'mean'
    ignore_index = np.int64(-1)

    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y', 'w'],
                                 outputs=['z', 'log_prob'],
                                 reduction=reduction,
                                 ignore_index=ignore_index)

    N, C, dim1 = 3, 5, 6
    np.random.seed(0)
    x = np.random.rand(N, C, dim1).astype(np.float32)
    labels = np.random.randint(0, high=C, size=(N, dim1)).astype(np.int64)
    labels[0][0] = -1
    weight = np.random.rand(C).astype(np.float32)

    loss, log_prob = softmaxcrossentropy(x,
                              labels,
                              weight=weight,
                              reduction=reduction,
                              ignore_index=ignore_index,
                              get_log_prob=True)

    expect(node, inputs=[x, labels, weight], outputs=[loss, log_prob], name='test_sce_NCd1_mean_weight_negative_ii_log_prob')

**input_shape_is_NCd1d2d3_none_no_weight_negative_ii**

::

    reduction = 'none'
    ignore_index = np.int64(-5)

    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y'],
                                 outputs=['z'],
                                 reduction=reduction,
                                 ignore_index=ignore_index)

    N, C, dim1, dim2, dim3 = 3, 5, 6, 6, 5
    np.random.seed(0)
    x = np.random.rand(N, C, dim1, dim2, dim3).astype(np.float32)
    labels = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3)).astype(np.int64)
    labels[0][0][0][0] = -5

    sce = softmaxcrossentropy(x,
                              labels,
                              reduction=reduction,
                              ignore_index=ignore_index)

    expect(node, inputs=[x, labels], outputs=[sce], name='test_sce_NCd1d2d3_none_no_weight_negative_ii')

**input_shape_is_NCd1d2d3_none_no_weight_negative_ii_log_prob**

::

    reduction = 'none'
    ignore_index = np.int64(-5)

    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y'],
                                 outputs=['z', 'log_prob'],
                                 reduction=reduction,
                                 ignore_index=ignore_index)

    N, C, dim1, dim2, dim3 = 3, 5, 6, 6, 5
    np.random.seed(0)
    x = np.random.rand(N, C, dim1, dim2, dim3).astype(np.float32)
    labels = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3)).astype(np.int64)
    labels[0][0][0][0] = -5

    loss, log_prob = softmaxcrossentropy(x,
                              labels,
                              reduction=reduction,
                              ignore_index=ignore_index,
                              get_log_prob=True)

    expect(node, inputs=[x, labels], outputs=[loss, log_prob], name='test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob')

**input_shape_is_NCd1d2d3_sum_weight_high_ii**

::

    reduction = 'sum'
    ignore_index = np.int64(10)

    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y', 'w'],
                                 outputs=['z'],
                                 reduction=reduction,
                                 ignore_index=ignore_index)

    N, C = 3, 5
    np.random.seed(0)
    x = np.random.rand(N, C).astype(np.float32)
    labels = np.random.randint(0, high=C, size=(N)).astype(np.int64)
    labels[0] = 10
    weight = np.random.rand(C).astype(np.float32)

    sce = softmaxcrossentropy(x,
                              labels,
                              weight=weight,
                              reduction=reduction,
                              ignore_index=ignore_index)

    expect(node, inputs=[x, labels, weight], outputs=[sce], name='test_sce_NCd1d2d3_sum_weight_high_ii')

**input_shape_is_NCd1d2d3_sum_weight_high_ii_log_prob**

::

    reduction = 'sum'
    ignore_index = np.int64(10)

    node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                                 inputs=['x', 'y', 'w'],
                                 outputs=['z', 'log_prob'],
                                 reduction=reduction,
                                 ignore_index=ignore_index)

    N, C = 3, 5
    np.random.seed(0)
    x = np.random.rand(N, C).astype(np.float32)
    labels = np.random.randint(0, high=C, size=(N)).astype(np.int64)
    labels[0] = 10
    weight = np.random.rand(C).astype(np.float32)

    loss, log_prob = softmaxcrossentropy(x,
                              labels,
                              weight=weight,
                              reduction=reduction,
                              ignore_index=ignore_index,
                              get_log_prob=True)

    expect(node, inputs=[x, labels, weight], outputs=[loss, log_prob], name='test_sce_NCd1d2d3_sum_weight_high_ii_log_prob')
