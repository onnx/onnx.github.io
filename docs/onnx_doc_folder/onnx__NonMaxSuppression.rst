
.. _l-onnx-doc-NonMaxSuppression:

=================
NonMaxSuppression
=================

.. contents::
    :local:


.. _l-onnx-op-nonmaxsuppression-11:

NonMaxSuppression - 11
======================

**Version**

* **name**: `NonMaxSuppression (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#NonMaxSuppression>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.
Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to
orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system
result in the same boxes being selected by the algorithm.
The selected_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.
The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.

**Attributes**

* **center_point_box**:
  Integer indicate the format of the box data. The default is 0. 0 -
  the box data is supplied as [y1, x1, y2, x2] where (y1, x1) and (y2,
  x2) are the coordinates of any diagonal pair of box corners and the
  coordinates can be provided as normalized (i.e., lying in the
  interval [0, 1]) or absolute. Mostly used for TF models. 1 - the box
  data is supplied as [x_center, y_center, width, height]. Mostly used
  for Pytorch models. Default value is ``0``.

**Inputs**

Between 2 and 5 inputs.

* **boxes** (heterogeneous) - **tensor(float)**:
  An input tensor with shape [num_batches, spatial_dimension, 4]. The
  single box data format is indicated by center_point_box.
* **scores** (heterogeneous) - **tensor(float)**:
  An input tensor with shape [num_batches, num_classes,
  spatial_dimension]
* **max_output_boxes_per_class** (optional, heterogeneous) - **tensor(int64)**:
  Integer representing the maximum number of boxes to be selected per
  batch per class. It is a scalar. Default to 0, which means no
  output.
* **iou_threshold** (optional, heterogeneous) - **tensor(float)**:
  Float representing the threshold for deciding whether boxes overlap
  too much with respect to IOU. It is scalar. Value range [0, 1].
  Default to 0.
* **score_threshold** (optional, heterogeneous) - **tensor(float)**:
  Float representing the threshold for deciding when to remove boxes
  based on score. It is a scalar.

**Outputs**

* **selected_indices** (heterogeneous) - **tensor(int64)**:
  selected indices from the boxes tensor. [num_selected_indices, 3],
  the selected index format is [batch_index, class_index, box_index].

**Examples**

**nonmaxsuppression_suppress_by_IOU**

::

    node = onnx.helper.make_node(
        'NonMaxSuppression',
        inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
        outputs=['selected_indices']
    )
    boxes = np.array([[
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.1, 1.0, 1.1],
        [0.0, -0.1, 1.0, 0.9],
        [0.0, 10.0, 1.0, 11.0],
        [0.0, 10.1, 1.0, 11.1],
        [0.0, 100.0, 1.0, 101.0]
    ]]).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)

    expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_suppress_by_IOU')

**nonmaxsuppression_suppress_by_IOU_and_scores**

::

    node = onnx.helper.make_node(
        'NonMaxSuppression',
        inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
        outputs=['selected_indices']
    )
    boxes = np.array([[
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.1, 1.0, 1.1],
        [0.0, -0.1, 1.0, 0.9],
        [0.0, 10.0, 1.0, 11.0],
        [0.0, 10.1, 1.0, 11.1],
        [0.0, 100.0, 1.0, 101.0]
    ]]).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.4]).astype(np.float32)
    selected_indices = np.array([[0, 0, 3], [0, 0, 0]]).astype(np.int64)

    expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_suppress_by_IOU_and_scores')

**nonmaxsuppression_flipped_coordinates**

::

    node = onnx.helper.make_node(
        'NonMaxSuppression',
        inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
        outputs=['selected_indices']
    )
    boxes = np.array([[
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.1, 1.0, 1.1],
        [0.0, 0.9, 1.0, -0.1],
        [0.0, 10.0, 1.0, 11.0],
        [1.0, 10.1, 0.0, 11.1],
        [1.0, 101.0, 0.0, 100.0]
    ]]).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)

    expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_flipped_coordinates')

**nonmaxsuppression_limit_output_size**

::

    node = onnx.helper.make_node(
        'NonMaxSuppression',
        inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
        outputs=['selected_indices']
    )
    boxes = np.array([[
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.1, 1.0, 1.1],
        [0.0, -0.1, 1.0, 0.9],
        [0.0, 10.0, 1.0, 11.0],
        [0.0, 10.1, 1.0, 11.1],
        [0.0, 100.0, 1.0, 101.0]
    ]]).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([2]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([[0, 0, 3], [0, 0, 0]]).astype(np.int64)

    expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_limit_output_size')

**nonmaxsuppression_single_box**

::

    node = onnx.helper.make_node(
        'NonMaxSuppression',
        inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
        outputs=['selected_indices']
    )
    boxes = np.array([[
        [0.0, 0.0, 1.0, 1.0]
    ]]).astype(np.float32)
    scores = np.array([[[0.9]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([[0, 0, 0]]).astype(np.int64)

    expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_single_box')

**nonmaxsuppression_identical_boxes**

::

    node = onnx.helper.make_node(
        'NonMaxSuppression',
        inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
        outputs=['selected_indices']
    )
    boxes = np.array([[
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],

        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0]
    ]]).astype(np.float32)
    scores = np.array([[[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([[0, 0, 0]]).astype(np.int64)

    expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_identical_boxes')

**nonmaxsuppression_center_point_box_format**

::

    node = onnx.helper.make_node(
        'NonMaxSuppression',
        inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
        outputs=['selected_indices'],
        center_point_box=1
    )
    boxes = np.array([[
        [0.5, 0.5, 1.0, 1.0],
        [0.5, 0.6, 1.0, 1.0],
        [0.5, 0.4, 1.0, 1.0],
        [0.5, 10.5, 1.0, 1.0],
        [0.5, 10.6, 1.0, 1.0],
        [0.5, 100.5, 1.0, 1.0]
    ]]).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)

    expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_center_point_box_format')

**nonmaxsuppression_two_classes**

::

    node = onnx.helper.make_node(
        'NonMaxSuppression',
        inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
        outputs=['selected_indices']
    )
    boxes = np.array([[
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.1, 1.0, 1.1],
        [0.0, -0.1, 1.0, 0.9],
        [0.0, 10.0, 1.0, 11.0],
        [0.0, 10.1, 1.0, 11.1],
        [0.0, 100.0, 1.0, 101.0]
    ]]).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
                        [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([2]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 1, 3], [0, 1, 0]]).astype(np.int64)

    expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_two_classes')

**nonmaxsuppression_two_batches**

::

    node = onnx.helper.make_node(
        'NonMaxSuppression',
        inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
        outputs=['selected_indices']
    )
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0],
                       [0.0, 0.1, 1.0, 1.1],
                       [0.0, -0.1, 1.0, 0.9],
                       [0.0, 10.0, 1.0, 11.0],
                       [0.0, 10.1, 1.0, 11.1],
                       [0.0, 100.0, 1.0, 101.0]],
                      [[0.0, 0.0, 1.0, 1.0],
                       [0.0, 0.1, 1.0, 1.1],
                       [0.0, -0.1, 1.0, 0.9],
                       [0.0, 10.0, 1.0, 11.0],
                       [0.0, 10.1, 1.0, 11.1],
                       [0.0, 100.0, 1.0, 101.0]]]).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]],
                       [[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([2]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)
    selected_indices = np.array([[0, 0, 3], [0, 0, 0], [1, 0, 3], [1, 0, 0]]).astype(np.int64)

    expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_two_batches')

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">result in the same boxes being selected by the algorithm.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">result in the same boxes being selected by the algorithm.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The selected_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The selected_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **center_point_box**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **center_point_box**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Integer indicate the format of the box data. The default is 0. 0 -</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Integer indicate the format of the box data. The default is 0. 0 -</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the box data is supplied as [y1, x1, y2, x2] where (y1, x1) and (y2,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the box data is supplied as [y1, x1, y2, x2] where (y1, x1) and (y2,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x2) are the coordinates of any diagonal pair of box corners and the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  x2) are the coordinates of any diagonal pair of box corners and the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  coordinates can be provided as normalized (i.e., lying in the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  coordinates can be provided as normalized (i.e., lying in the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  interval [0, 1]) or absolute. Mostly used for TF models. 1 - the box</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  interval [0, 1]) or absolute. Mostly used for TF models. 1 - the box</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data is supplied as [x_center, y_center, width, height]. Mostly used</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  data is supplied as [x_center, y_center, width, height]. Mostly used</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  for Pytorch models. Default value is 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  for Pytorch models. Default value is 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 5 inputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 5 inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **boxes** (heterogeneous) - **tensor(float)**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **boxes** (heterogeneous) - **tensor(float)**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  An input tensor with shape [num_batches, spatial_dimension, 4]. The</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  An input tensor with shape [num_batches, spatial_dimension, 4]. The</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  single box data format is indicated by center_point_box.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  single box data format is indicated by center_point_box.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **scores** (heterogeneous) - **tensor(float)**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **scores** (heterogeneous) - **tensor(float)**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  An input tensor with shape [num_batches, num_classes,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  An input tensor with shape [num_batches, num_classes,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  spatial_dimension]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  spatial_dimension]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **max_output_boxes_per_class** (optional, heterogeneous) - **tensor(int64)**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **max_output_boxes_per_class** (optional, heterogeneous) - **tensor(int64)**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Integer representing the maximum number of boxes to be selected per</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Integer representing the maximum number of boxes to be selected per</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch per class. It is a scalar. Default to 0, which means no</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  batch per class. It is a scalar. Default to 0, which means no</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **iou_threshold** (optional, heterogeneous) - **tensor(float)**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **iou_threshold** (optional, heterogeneous) - **tensor(float)**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Float representing the threshold for deciding whether boxes overlap</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Float representing the threshold for deciding whether boxes overlap</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  too much with respect to IOU. It is scalar. Value range [0, 1].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  too much with respect to IOU. It is scalar. Value range [0, 1].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Default to 0.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Default to 0.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **score_threshold** (optional, heterogeneous) - **tensor(float)**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **score_threshold** (optional, heterogeneous) - **tensor(float)**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Float representing the threshold for deciding when to remove boxes</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Float representing the threshold for deciding when to remove boxes</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  based on score. It is a scalar.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  based on score. It is a scalar.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **selected_indices** (heterogeneous) - **tensor(int64)**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **selected_indices** (heterogeneous) - **tensor(int64)**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  selected indices from the boxes tensor. [num_selected_indices, 3],</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  selected indices from the boxes tensor. [num_selected_indices, 3],</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the selected index format is [batch_index, class_index, box_index].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the selected index format is [batch_index, class_index, box_index].</code></td></tr>
    </table>

.. _l-onnx-op-nonmaxsuppression-10:

NonMaxSuppression - 10
======================

**Version**

* **name**: `NonMaxSuppression (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#NonMaxSuppression>`_
* **domain**: **main**
* **since_version**: **10**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 10**.

**Summary**

Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
Bounding boxes with score less than score_threshold are removed. Bounding box format is indicated by attribute center_point_box.
Note that this algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to
orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system
result in the same boxes being selected by the algorithm.
The selected_indices output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes.
The bounding box coordinates corresponding to the selected indices can then be obtained using the Gather or GatherND operation.

**Attributes**

* **center_point_box**:
  Integer indicate the format of the box data. The default is 0. 0 -
  the box data is supplied as [y1, x1, y2, x2] where (y1, x1) and (y2,
  x2) are the coordinates of any diagonal pair of box corners and the
  coordinates can be provided as normalized (i.e., lying in the
  interval [0, 1]) or absolute. Mostly used for TF models. 1 - the box
  data is supplied as [x_center, y_center, width, height]. Mostly used
  for Pytorch models. Default value is ``0``.

**Inputs**

Between 2 and 5 inputs.

* **boxes** (heterogeneous) - **tensor(float)**:
  An input tensor with shape [num_batches, spatial_dimension, 4]. The
  single box data format is indicated by center_point_box.
* **scores** (heterogeneous) - **tensor(float)**:
  An input tensor with shape [num_batches, num_classes,
  spatial_dimension]
* **max_output_boxes_per_class** (optional, heterogeneous) - **tensor(int64)**:
  Integer representing the maximum number of boxes to be selected per
  batch per class. It is a scalar. Default to 0, which means no
  output.
* **iou_threshold** (optional, heterogeneous) - **tensor(float)**:
  Float representing the threshold for deciding whether boxes overlap
  too much with respect to IOU. It is scalar. Value range [0, 1].
  Default to 0.
* **score_threshold** (optional, heterogeneous) - **tensor(float)**:
  Float representing the threshold for deciding when to remove boxes
  based on score. It is a scalar.

**Outputs**

* **selected_indices** (heterogeneous) - **tensor(int64)**:
  selected indices from the boxes tensor. [num_selected_indices, 3],
  the selected index format is [batch_index, class_index, box_index].
