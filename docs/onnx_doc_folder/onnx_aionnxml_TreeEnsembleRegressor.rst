
.. _l-onnx-docai.onnx.ml-TreeEnsembleRegressor:

==================================
ai.onnx.ml - TreeEnsembleRegressor
==================================

.. contents::
    :local:


.. _l-onnx-opai-onnx-ml-treeensembleregressor-3:

TreeEnsembleRegressor - 3 (ai.onnx.ml)
======================================

**Version**

* **name**: `TreeEnsembleRegressor (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.TreeEnsembleRegressor>`_
* **domain**: **ai.onnx.ml**
* **since_version**: **3**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 3 of domain ai.onnx.ml**.

**Summary**

Tree Ensemble regressor.  Returns the regressed values for each input in N.

All args with nodes_ are fields of a tuple of tree nodes, and
it is assumed they are the same length, and an index i will decode the
tuple across these inputs.  Each node id can appear only once
for each tree id.

All fields prefixed with target_ are tuples of votes at the leaves.

A leaf may have multiple votes, where each vote is weighted by
the associated target_weights index.

All fields ending with <i>_as_tensor</i> can be used instead of the
same parameter without the suffix if the element type is double and not float.
All trees must have their node ids start at 0 and increment by 1.

Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF

**Attributes**

* **aggregate_function**:
  Defines how to aggregate leaf values within a target. <br>One of
  'AVERAGE,' 'SUM,' 'MIN,' 'MAX.' Default value is ``'SUM'``.
* **base_values**:
  Base values for classification, added to final class score; the size
  must be the same as the classes or can be left unassigned (assumed
  0)
* **base_values_as_tensor**:
  Base values for classification, added to final class score; the size
  must be the same as the classes or can be left unassigned (assumed
  0)
* **n_targets**:
  The total number of targets.
* **nodes_falsenodeids**:
  Child node if expression is false
* **nodes_featureids**:
  Feature id for each node.
* **nodes_hitrates**:
  Popularity of each node, used for performance and may be omitted.
* **nodes_hitrates_as_tensor**:
  Popularity of each node, used for performance and may be omitted.
* **nodes_missing_value_tracks_true**:
  For each node, define what to do in the presence of a NaN: use the
  'true' (if the attribute value is 1) or 'false' (if the attribute
  value is 0) branch based on the value in this array.<br>This
  attribute may be left undefined and the defalt value is false (0)
  for all nodes.
* **nodes_modes**:
  The node kind, that is, the comparison to make at the node. There is
  no comparison to make at a leaf node.<br>One of 'BRANCH_LEQ',
  'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ',
  'LEAF'
* **nodes_nodeids**:
  Node id for each node. Node ids must restart at zero for each tree
  and increase sequentially.
* **nodes_treeids**:
  Tree id for each node.
* **nodes_truenodeids**:
  Child node if expression is true
* **nodes_values**:
  Thresholds to do the splitting on for each node.
* **nodes_values_as_tensor**:
  Thresholds to do the splitting on for each node.
* **post_transform**:
  Indicates the transform to apply to the score. <br>One of 'NONE,'
  'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT' Default value is ``'NONE'``.
* **target_ids**:
  The index of the target that each weight is for
* **target_nodeids**:
  The node id of each weight
* **target_treeids**:
  The id of the tree that each node is in.
* **target_weights**:
  The weight for each target
* **target_weights_as_tensor**:
  The weight for each target

**Inputs**

* **X** (heterogeneous) - **T**:
  Input of shape [N,F]

**Outputs**

* **Y** (heterogeneous) - **tensor(float)**:
  N classes

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(int32),
  tensor(int64)
  ):
  The input type must be a tensor of a numeric type.

**Examples**

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Tree Ensemble regressor.  Returns the regressed values for each input in N.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Tree Ensemble regressor.  Returns the regressed values for each input in N.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">All args with nodes_ are fields of a tuple of tree nodes, and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">All args with nodes_ are fields of a tuple of tree nodes, and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">it is assumed they are the same length, and an index i will decode the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">it is assumed they are the same length, and an index i will decode the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">tuple across these inputs.  Each node id can appear only once</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">tuple across these inputs.  Each node id can appear only once</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">for each tree id.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">for each tree id.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">All fields prefixed with target_ are tuples of votes at the leaves.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">All fields prefixed with target_ are tuples of votes at the leaves.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">A leaf may have multiple votes, where each vote is weighted by</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">A leaf may have multiple votes, where each vote is weighted by</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the associated target_weights index.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the associated target_weights index.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">12</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">All fields ending with <i>_as_tensor</i> can be used instead of the</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">13</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">same parameter without the suffix if the element type is double and not float.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">All trees must have their node ids start at 0 and increment by 1.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">All trees must have their node ids start at 0 and increment by 1.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **aggregate_function**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **aggregate_function**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Defines how to aggregate leaf values within a target. <br>One of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Defines how to aggregate leaf values within a target. <br>One of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'AVERAGE,' 'SUM,' 'MIN,' 'MAX.' Default value is 'SUM'.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'AVERAGE,' 'SUM,' 'MIN,' 'MAX.' Default value is 'SUM'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **base_values**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **base_values**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Base values for classification, added to final class score; the size</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Base values for classification, added to final class score; the size</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  must be the same as the classes or can be left unassigned (assumed</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  must be the same as the classes or can be left unassigned (assumed</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  0)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  0)</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">27</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **base_values_as_tensor**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">28</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Base values for classification, added to final class score; the size</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">29</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  must be the same as the classes or can be left unassigned (assumed</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">30</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  0)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **n_targets**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **n_targets**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The total number of targets.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The total number of targets.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_falsenodeids**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_falsenodeids**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Child node if expression is false</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Child node if expression is false</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_featureids**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_featureids**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Feature id for each node.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Feature id for each node.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_hitrates**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_hitrates**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Popularity of each node, used for performance and may be omitted.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Popularity of each node, used for performance and may be omitted.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">39</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **nodes_hitrates_as_tensor**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">40</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Popularity of each node, used for performance and may be omitted.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_missing_value_tracks_true**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_missing_value_tracks_true**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  For each node, define what to do in the presence of a NaN: use the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  For each node, define what to do in the presence of a NaN: use the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'true' (if the attribute value is 1) or 'false' (if the attribute</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'true' (if the attribute value is 1) or 'false' (if the attribute</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  value is 0) branch based on the value in this array.<br>This</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  value is 0) branch based on the value in this array.<br>This</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  attribute may be left undefined and the defalt value is false (0)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  attribute may be left undefined and the defalt value is false (0)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  for all nodes.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  for all nodes.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_modes**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_modes**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The node kind, that is, the comparison to make at the node. There is</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The node kind, that is, the comparison to make at the node. There is</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  no comparison to make at a leaf node.<br>One of 'BRANCH_LEQ',</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  no comparison to make at a leaf node.<br>One of 'BRANCH_LEQ',</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ',</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ',</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'LEAF'</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'LEAF'</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_nodeids**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_nodeids**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Node id for each node. Node ids must restart at zero for each tree</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Node id for each node. Node ids must restart at zero for each tree</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  and increase sequentially.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  and increase sequentially.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_treeids**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_treeids**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tree id for each node.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tree id for each node.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_truenodeids**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_truenodeids**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Child node if expression is true</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Child node if expression is true</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_values**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_values**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Thresholds to do the splitting on for each node.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Thresholds to do the splitting on for each node.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">61</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **nodes_values_as_tensor**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">62</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Thresholds to do the splitting on for each node.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **post_transform**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **post_transform**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Indicates the transform to apply to the score. <br>One of 'NONE,'</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Indicates the transform to apply to the score. <br>One of 'NONE,'</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT' Default value is 'NONE'.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT' Default value is 'NONE'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **target_ids**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **target_ids**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The index of the target that each weight is for</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The index of the target that each weight is for</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **target_nodeids**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **target_nodeids**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The node id of each weight</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The node id of each weight</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **target_treeids**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **target_treeids**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The id of the tree that each node is in.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The id of the tree that each node is in.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **target_weights**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **target_weights**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The weight for each target</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The weight for each target</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">74</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **target_weights_as_tensor**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">75</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  The weight for each target</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input of shape [N,F]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input of shape [N,F]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **tensor(float)**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **tensor(float)**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  N classes</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  N classes</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input type must be a tensor of a numeric type.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input type must be a tensor of a numeric type.</code></td></tr>
    </table>

.. _l-onnx-opai-onnx-ml-treeensembleregressor-1:

TreeEnsembleRegressor - 1 (ai.onnx.ml)
======================================

**Version**

* **name**: `TreeEnsembleRegressor (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.TreeEnsembleRegressor>`_
* **domain**: **ai.onnx.ml**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: False

This version of the operator has been available
**since version 1 of domain ai.onnx.ml**.

**Summary**

Tree Ensemble regressor.  Returns the regressed values for each input in N.

All args with nodes_ are fields of a tuple of tree nodes, and
it is assumed they are the same length, and an index i will decode the
tuple across these inputs.  Each node id can appear only once
for each tree id.

All fields prefixed with target_ are tuples of votes at the leaves.

A leaf may have multiple votes, where each vote is weighted by
the associated target_weights index.

All trees must have their node ids start at 0 and increment by 1.

Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF

**Attributes**

* **aggregate_function**:
  Defines how to aggregate leaf values within a target. <br>One of
  'AVERAGE,' 'SUM,' 'MIN,' 'MAX.' Default value is ``'SUM'``.
* **base_values**:
  Base values for classification, added to final class score; the size
  must be the same as the classes or can be left unassigned (assumed
  0)
* **n_targets**:
  The total number of targets.
* **nodes_falsenodeids**:
  Child node if expression is false
* **nodes_featureids**:
  Feature id for each node.
* **nodes_hitrates**:
  Popularity of each node, used for performance and may be omitted.
* **nodes_missing_value_tracks_true**:
  For each node, define what to do in the presence of a NaN: use the
  'true' (if the attribute value is 1) or 'false' (if the attribute
  value is 0) branch based on the value in this array.<br>This
  attribute may be left undefined and the defalt value is false (0)
  for all nodes.
* **nodes_modes**:
  The node kind, that is, the comparison to make at the node. There is
  no comparison to make at a leaf node.<br>One of 'BRANCH_LEQ',
  'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ',
  'LEAF'
* **nodes_nodeids**:
  Node id for each node. Node ids must restart at zero for each tree
  and increase sequentially.
* **nodes_treeids**:
  Tree id for each node.
* **nodes_truenodeids**:
  Child node if expression is true
* **nodes_values**:
  Thresholds to do the splitting on for each node.
* **post_transform**:
  Indicates the transform to apply to the score. <br>One of 'NONE,'
  'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT' Default value is ``'NONE'``.
* **target_ids**:
  The index of the target that each weight is for
* **target_nodeids**:
  The node id of each weight
* **target_treeids**:
  The id of the tree that each node is in.
* **target_weights**:
  The weight for each target

**Inputs**

* **X** (heterogeneous) - **T**:
  Input of shape [N,F]

**Outputs**

* **Y** (heterogeneous) - **tensor(float)**:
  N classes

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(int32),
  tensor(int64)
  ):
  The input type must be a tensor of a numeric type.
