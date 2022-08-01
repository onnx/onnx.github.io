
.. _l-onnx-docai.onnx.ml-TreeEnsembleClassifier:

===================================
ai.onnx.ml - TreeEnsembleClassifier
===================================

.. contents::
    :local:


.. _l-onnx-opai-onnx-ml-treeensembleclassifier-3:

TreeEnsembleClassifier - 3 (ai.onnx.ml)
=======================================

**Version**

* **name**: `TreeEnsembleClassifier (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.TreeEnsembleClassifier>`_
* **domain**: **ai.onnx.ml**
* **since_version**: **3**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 3 of domain ai.onnx.ml**.

**Summary**

Tree Ensemble classifier. Returns the top class for each of N inputs.

The attributes named 'nodes_X' form a sequence of tuples, associated by
index into the sequences, which must all be of equal length. These tuples
define the nodes.

Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.
A leaf may have multiple votes, where each vote is weighted by
the associated class_weights index.

One and only one of classlabels_strings or classlabels_int64s
will be defined. The class_ids are indices into this list.
All fields ending with <i>_as_tensor</i> can be used instead of the
same parameter without the suffix if the element type is double and not float.

**Attributes**

* **base_values**:
  Base values for classification, added to final class score; the size
  must be the same as the classes or can be left unassigned (assumed
  0)
* **base_values_as_tensor**:
  Base values for classification, added to final class score; the size
  must be the same as the classes or can be left unassigned (assumed
  0)
* **class_ids**:
  The index of the class list that each weight is for.
* **class_nodeids**:
  node id that this weight is for.
* **class_treeids**:
  The id of the tree that this node is in.
* **class_weights**:
  The weight for the class in class_id.
* **class_weights_as_tensor**:
  The weight for the class in class_id.
* **classlabels_int64s**:
  Class labels if using integer labels.<br>One and only one of the
  'classlabels_*' attributes must be defined.
* **classlabels_strings**:
  Class labels if using string labels.<br>One and only one of the
  'classlabels_*' attributes must be defined.
* **nodes_falsenodeids**:
  Child node if expression is false.
* **nodes_featureids**:
  Feature id for each node.
* **nodes_hitrates**:
  Popularity of each node, used for performance and may be omitted.
* **nodes_hitrates_as_tensor**:
  Popularity of each node, used for performance and may be omitted.
* **nodes_missing_value_tracks_true**:
  For each node, define what to do in the presence of a missing value:
  if a value is missing (NaN), use the 'true' or 'false' branch based
  on the value in this array.<br>This attribute may be left undefined,
  and the defalt value is false (0) for all nodes.
* **nodes_modes**:
  The node kind, that is, the comparison to make at the node. There is
  no comparison to make at a leaf node.<br>One of 'BRANCH_LEQ',
  'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ',
  'LEAF'
* **nodes_nodeids**:
  Node id for each node. Ids may restart at zero for each tree, but it
  not required to.
* **nodes_treeids**:
  Tree id for each node.
* **nodes_truenodeids**:
  Child node if expression is true.
* **nodes_values**:
  Thresholds to do the splitting on for each node.
* **nodes_values_as_tensor**:
  Thresholds to do the splitting on for each node.
* **post_transform**:
  Indicates the transform to apply to the score. <br> One of 'NONE,'
  'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT.' Default value is ``'NONE'``.

**Inputs**

* **X** (heterogeneous) - **T1**:
  Input of shape [N,F]

**Outputs**

* **Y** (heterogeneous) - **T2**:
  N, Top class for each point
* **Z** (heterogeneous) - **tensor(float)**:
  The class score for each class, for each point, a tensor of shape
  [N,E].

**Type Constraints**

* **T1** in (
  tensor(double),
  tensor(float),
  tensor(int32),
  tensor(int64)
  ):
  The input type must be a tensor of a numeric type.
* **T2** in (
  tensor(int64),
  tensor(string)
  ):
  The output type will be a tensor of strings or integers, depending
  on which of the classlabels_* attributes is used.

**Examples**

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td><code>0</code></td><td><code>0</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">Tree Ensemble classifier. <span style="color:#BA4A00;"> </span>Returns the top class for each of N inputs.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>Tree Ensemble classifier. Returns the top class for each of N inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The attributes named 'nodes_X' form a sequence of tuples, associated by</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The attributes named 'nodes_X' form a sequence of tuples, associated by</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">index into the sequences, which must all be of equal length. These tuples</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">index into the sequences, which must all be of equal length. These tuples</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">define the nodes.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">define the nodes.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">A leaf may have multiple votes, where each vote is weighted by</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">A leaf may have multiple votes, where each vote is weighted by</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the associated class_weights index.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the associated class_weights index.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">One and only one of classlabels_strings or classlabels_int64s</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">One and only one of classlabels_strings or classlabels_int64s</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">will be defined. The class_ids are indices into this list.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">will be defined. The class_ids are indices into this list.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">12</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">All fields ending with <i>_as_tensor</i> can be used instead of the</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">13</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">same parameter without the suffix if the element type is double and not float.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **base_values**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **base_values**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Base values for classification, added to final class score; the size</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Base values for classification, added to final class score; the size</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  must be the same as the classes or can be left unassigned (assumed</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  must be the same as the classes or can be left unassigned (assumed</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  0)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  0)</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">21</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **base_values_as_tensor**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">22</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Base values for classification, added to final class score; the size</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">23</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  must be the same as the classes or can be left unassigned (assumed</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">24</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  0)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **class_ids**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **class_ids**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The index of the class list that each weight is for.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The index of the class list that each weight is for.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **class_nodeids**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **class_nodeids**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  node id that this weight is for.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  node id that this weight is for.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **class_treeids**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **class_treeids**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The id of the tree that this node is in.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The id of the tree that this node is in.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **class_weights**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **class_weights**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The weight for the class in class_id.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The weight for the class in class_id.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">33</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **class_weights_as_tensor**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">34</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  The weight for the class in class_id.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **classlabels_int64s**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **classlabels_int64s**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Class labels if using integer labels.<br>One and only one of the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Class labels if using integer labels.<br>One and only one of the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'classlabels_*' attributes must be defined.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'classlabels_*' attributes must be defined.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **classlabels_strings**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **classlabels_strings**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Class labels if using string labels.<br>One and only one of the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Class labels if using string labels.<br>One and only one of the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'classlabels_*' attributes must be defined.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'classlabels_*' attributes must be defined.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_falsenodeids**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_falsenodeids**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Child node if expression is false.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Child node if expression is false.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_featureids**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_featureids**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Feature id for each node.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Feature id for each node.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_hitrates**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_hitrates**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Popularity of each node, used for performance and may be omitted.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Popularity of each node, used for performance and may be omitted.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">47</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **nodes_hitrates_as_tensor**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">48</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Popularity of each node, used for performance and may be omitted.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_missing_value_tracks_true**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_missing_value_tracks_true**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  For each node, define what to do in the presence of a missing value:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  For each node, define what to do in the presence of a missing value:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  if a value is missing (NaN), use the 'true' or 'false' branch based</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  if a value is missing (NaN), use the 'true' or 'false' branch based</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  on the value in this array.<br>This attribute may be left undefined,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  on the value in this array.<br>This attribute may be left undefined,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  and the defalt value is false (0) for all nodes.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  and the defalt value is false (0) for all nodes.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_modes**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_modes**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The node kind, that is, the comparison to make at the node. There is</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The node kind, that is, the comparison to make at the node. There is</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  no comparison to make at a leaf node.<br>One of 'BRANCH_LEQ',</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  no comparison to make at a leaf node.<br>One of 'BRANCH_LEQ',</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ',</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ',</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'LEAF'</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'LEAF'</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_nodeids**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_nodeids**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Node id for each node. Ids may restart at zero for each tree, but it</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Node id for each node. Ids may restart at zero for each tree, but it</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  not required to.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  not required to.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_treeids**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_treeids**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tree id for each node.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Tree id for each node.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_truenodeids**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_truenodeids**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Child node if expression is true.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Child node if expression is true.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_values**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **nodes_values**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Thresholds to do the splitting on for each node.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Thresholds to do the splitting on for each node.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">68</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">* **nodes_values_as_tensor**:</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">69</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Thresholds to do the splitting on for each node.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **post_transform**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **post_transform**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Indicates the transform to apply to the score. <br> One of 'NONE,'</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Indicates the transform to apply to the score. <br> One of 'NONE,'</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT.' Default value is 'NONE'.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT.' Default value is 'NONE'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T1**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **X** (heterogeneous) - **T1**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input of shape [N,F]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Input of shape [N,F]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T2**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Y** (heterogeneous) - **T2**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  N, Top class for each point</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  N, Top class for each point</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Z** (heterogeneous) - **tensor(float)**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Z** (heterogeneous) - **tensor(float)**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The class score for each class, for each point, a tensor of shape</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The class score for each class, for each point, a tensor of shape</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  [N,E].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  [N,E].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T1** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input type must be a tensor of a numeric type.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The input type must be a tensor of a numeric type.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T2** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T2** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output type will be a tensor of strings or integers, depending</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The output type will be a tensor of strings or integers, depending</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  on which of the classlabels_* attributes is used.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  on which of the classlabels_* attributes is used.</code></td></tr>
    </table>

.. _l-onnx-opai-onnx-ml-treeensembleclassifier-1:

TreeEnsembleClassifier - 1 (ai.onnx.ml)
=======================================

**Version**

* **name**: `TreeEnsembleClassifier (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.TreeEnsembleClassifier>`_
* **domain**: **ai.onnx.ml**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1 of domain ai.onnx.ml**.

**Summary**

Tree Ensemble classifier.  Returns the top class for each of N inputs.

The attributes named 'nodes_X' form a sequence of tuples, associated by
index into the sequences, which must all be of equal length. These tuples
define the nodes.

Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.
A leaf may have multiple votes, where each vote is weighted by
the associated class_weights index.

One and only one of classlabels_strings or classlabels_int64s
will be defined. The class_ids are indices into this list.

**Attributes**

* **base_values**:
  Base values for classification, added to final class score; the size
  must be the same as the classes or can be left unassigned (assumed
  0)
* **class_ids**:
  The index of the class list that each weight is for.
* **class_nodeids**:
  node id that this weight is for.
* **class_treeids**:
  The id of the tree that this node is in.
* **class_weights**:
  The weight for the class in class_id.
* **classlabels_int64s**:
  Class labels if using integer labels.<br>One and only one of the
  'classlabels_*' attributes must be defined.
* **classlabels_strings**:
  Class labels if using string labels.<br>One and only one of the
  'classlabels_*' attributes must be defined.
* **nodes_falsenodeids**:
  Child node if expression is false.
* **nodes_featureids**:
  Feature id for each node.
* **nodes_hitrates**:
  Popularity of each node, used for performance and may be omitted.
* **nodes_missing_value_tracks_true**:
  For each node, define what to do in the presence of a missing value:
  if a value is missing (NaN), use the 'true' or 'false' branch based
  on the value in this array.<br>This attribute may be left undefined,
  and the defalt value is false (0) for all nodes.
* **nodes_modes**:
  The node kind, that is, the comparison to make at the node. There is
  no comparison to make at a leaf node.<br>One of 'BRANCH_LEQ',
  'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ',
  'LEAF'
* **nodes_nodeids**:
  Node id for each node. Ids may restart at zero for each tree, but it
  not required to.
* **nodes_treeids**:
  Tree id for each node.
* **nodes_truenodeids**:
  Child node if expression is true.
* **nodes_values**:
  Thresholds to do the splitting on for each node.
* **post_transform**:
  Indicates the transform to apply to the score. <br> One of 'NONE,'
  'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT.' Default value is ``'NONE'``.

**Inputs**

* **X** (heterogeneous) - **T1**:
  Input of shape [N,F]

**Outputs**

* **Y** (heterogeneous) - **T2**:
  N, Top class for each point
* **Z** (heterogeneous) - **tensor(float)**:
  The class score for each class, for each point, a tensor of shape
  [N,E].

**Type Constraints**

* **T1** in (
  tensor(double),
  tensor(float),
  tensor(int32),
  tensor(int64)
  ):
  The input type must be a tensor of a numeric type.
* **T2** in (
  tensor(int64),
  tensor(string)
  ):
  The output type will be a tensor of strings or integers, depending
  on which of the classlabels_* attributes is used.
