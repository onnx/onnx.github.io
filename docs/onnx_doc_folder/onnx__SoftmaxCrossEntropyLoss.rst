
.. _l-onnx-doc-SoftmaxCrossEntropyLoss:

=======================
SoftmaxCrossEntropyLoss
=======================

.. contents::
    :local:


.. _l-onnx-op-softmaxcrossentropyloss-13:

SoftmaxCrossEntropyLoss - 13
============================

**Version**

* **name**: `SoftmaxCrossEntropyLoss (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#SoftmaxCrossEntropyLoss>`_
* **domain**: **main**
* **since_version**: **13**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 13**.

**Summary**

Loss function that measures the softmax cross entropy
between 'scores' and 'labels'.
This operator first computes a loss tensor whose shape is identical to the labels input.
If the input is 2-D with shape (N, C), the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N).
If the input is N-D tensor with shape (N, C, D1, D2, ..., Dk),
the loss tensor L may have (N, D1, D2, ..., Dk) as its shape and L[i,][j_1][j_2]...[j_k] denotes a scalar element in L.
After L is available, this operator can optionally do a reduction operator.

shape(scores): (N, C) where C is the number of classes, or (N, C, D1, D2,..., Dk),
        with K >= 1 in case of K-dimensional loss.
shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N, D1, D2,..., Dk),
        with K >= 1 in case of K-dimensional loss.

The loss for one sample, l_i, can caculated as follows:
    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk], where i is the index of classes.
or
    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk] * weights[c], if 'weights' is provided.

loss is zero for the case when label-value equals ignore_index.
    l[i][d1][d2]...[dk]  = 0, when labels[n][d1][d2]...[dk] = ignore_index

where:
    p = Softmax(scores)
    y = Log(p)
    c = labels[i][d1][d2]...[dk]

Finally, L is optionally reduced:
If reduction = 'none', the output is L with shape (N, D1, D2, ..., Dk).
If reduction = 'sum', the output is scalar: Sum(L).
If reduction = 'mean', the output is scalar: ReduceMean(L), or if weight is provided: ReduceSum(L) / ReduceSum(W),
where tensor W is of shape (N, D1, D2, ..., Dk) and W[n][d1][d2]...[dk] = weights[labels[i][d1][d2]...[dk]].

**Attributes**

* **ignore_index**:
  Specifies a target value that is ignored and does not contribute to
  the input gradient. It's an optional value.
* **reduction**:
  Type of reduction to apply to loss: none, sum, mean(default).
  'none': no reduction will be applied, 'sum': the output will be
  summed. 'mean': the sum of the output will be divided by the number
  of elements in the output. Default value is ``'mean'``.

**Inputs**

Between 2 and 3 inputs.

* **scores** (heterogeneous) - **T**:
  The predicted outputs with shape [batch_size, class_size], or
  [batch_size, class_size, D1, D2 , ..., Dk], where K is the number of
  dimensions.
* **labels** (heterogeneous) - **Tind**:
  The ground truth output tensor, with shape [batch_size], or
  [batch_size, D1, D2, ..., Dk], where K is the number of dimensions.
  Labels element value shall be in range of [0, C). If ignore_index is
  specified, it may have a value outside [0, C) and the label values
  should either be in the range [0, C) or have the value ignore_index.
* **weights** (optional, heterogeneous) - **T**:
  A manual rescaling weight given to each class. If given, it has to
  be a 1D Tensor assigning weight to each of the classes. Otherwise,
  it is treated as if having all ones.

**Outputs**

Between 1 and 2 outputs.

* **output** (heterogeneous) - **T**:
  Weighted loss float Tensor. If reduction is 'none', this has the
  shape of [batch_size], or [batch_size, D1, D2, ..., Dk] in case of
  K-dimensional loss. Otherwise, it is a scalar.
* **log_prob** (optional, heterogeneous) - **T**:
  Log probability tensor. If the output of softmax is prob, its value
  is log(prob).

**Type Constraints**

* **T** in (
  tensor(bfloat16),
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
* **Tind** in (
  tensor(int32),
  tensor(int64)
  ):
  Constrain target to integer types

**Examples**

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Loss function that measures the softmax cross entropy</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Loss function that measures the softmax cross entropy</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">between 'scores' and 'labels'.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">between 'scores' and 'labels'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator first computes a loss tensor whose shape is identical to the labels input.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This operator first computes a loss tensor whose shape is identical to the labels input.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If the input is 2-D with shape (N, C), the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If the input is 2-D with shape (N, C), the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If the input is N-D tensor with shape (N, C, D1, D2, ..., Dk),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If the input is N-D tensor with shape (N, C, D1, D2, ..., Dk),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the loss tensor L may have (N, D1, D2, ..., Dk) as its shape and L[i,][j_1][j_2]...[j_k] denotes a scalar element in L.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the loss tensor L may have (N, D1, D2, ..., Dk) as its shape and L[i,][j_1][j_2]...[j_k] denotes a scalar element in L.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">After L is available, this operator can optionally do a reduction operator.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">After L is available, this operator can optionally do a reduction operator.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">shape(scores): (N, C) where C is the number of classes, or (N, C, D1, D2,..., Dk),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">shape(scores): (N, C) where C is the number of classes, or (N, C, D1, D2,..., Dk),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        with K >= 1 in case of K-dimensional loss.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        with K >= 1 in case of K-dimensional loss.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N, D1, D2,..., Dk),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N, D1, D2,..., Dk),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        with K >= 1 in case of K-dimensional loss.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        with K >= 1 in case of K-dimensional loss.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The loss for one sample, l_i, can caculated as follows:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The loss for one sample, l_i, can caculated as follows:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk], where i is the index of classes.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk], where i is the index of classes.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">or</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">or</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk] * weights[c], if 'weights' is provided.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk] * weights[c], if 'weights' is provided.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">loss is zero for the case when label-value equals ignore_index.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">loss is zero for the case when label-value equals ignore_index.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    l[i][d1][d2]...[dk]  = 0, when labels[n][d1][d2]...[dk] = ignore_index</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    l[i][d1][d2]...[dk]  = 0, when labels[n][d1][d2]...[dk] = ignore_index</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">where:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">where:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    p = Softmax(scores)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    p = Softmax(scores)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    y = Log(p)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    y = Log(p)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    c = labels[i][d1][d2]...[dk]</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    c = labels[i][d1][d2]...[dk]</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Finally, L is optionally reduced:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Finally, L is optionally reduced:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If reduction = 'none', the output is L with shape (N, D1, D2, ..., Dk).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If reduction = 'none', the output is L with shape (N, D1, D2, ..., Dk).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If reduction = 'sum', the output is scalar: Sum(L).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If reduction = 'sum', the output is scalar: Sum(L).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If reduction = 'mean', the output is scalar: ReduceMean(L), or if weight is provided: ReduceSum(L) / ReduceSum(W),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">If reduction = 'mean', the output is scalar: ReduceMean(L), or if weight is provided: ReduceSum(L) / ReduceSum(W),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">where tensor W is of shape (N, D1, D2, ..., Dk) and W[n][d1][d2]...[dk] = weights[labels[i][d1][d2]...[dk]].</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">where tensor W is of shape (N, D1, D2, ..., Dk) and W[n][d1][d2]...[dk] = weights[labels[i][d1][d2]...[dk]].</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **ignore_index**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **ignore_index**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Specifies a target value that is ignored and does not contribute to</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Specifies a target value that is ignored and does not contribute to</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the input gradient. It's an optional value.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  the input gradient. It's an optional value.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **reduction**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **reduction**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Type of reduction to apply to loss: none, sum, mean(default).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Type of reduction to apply to loss: none, sum, mean(default).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'none': no reduction will be applied, 'sum': the output will be</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  'none': no reduction will be applied, 'sum': the output will be</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  summed. 'mean': the sum of the output will be divided by the number</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  summed. 'mean': the sum of the output will be divided by the number</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  of elements in the output. Default value is 'mean'.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  of elements in the output. Default value is 'mean'.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 3 inputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 3 inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **scores** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **scores** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The predicted outputs with shape [batch_size, class_size], or</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The predicted outputs with shape [batch_size, class_size], or</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  [batch_size, class_size, D1, D2 , ..., Dk], where K is the number of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  [batch_size, class_size, D1, D2 , ..., Dk], where K is the number of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimensions.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  dimensions.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **labels** (heterogeneous) - **Tind**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **labels** (heterogeneous) - **Tind**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The ground truth output tensor, with shape [batch_size], or</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The ground truth output tensor, with shape [batch_size], or</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  [batch_size, D1, D2, ..., Dk], where K is the number of dimensions.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  [batch_size, D1, D2, ..., Dk], where K is the number of dimensions.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Labels element value shall be in range of [0, C). If ignore_index is</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Labels element value shall be in range of [0, C). If ignore_index is</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  specified, it may have a value outside [0, C) and the label values</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  specified, it may have a value outside [0, C) and the label values</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  should either be in the range [0, C) or have the value ignore_index.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  should either be in the range [0, C) or have the value ignore_index.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **weights** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **weights** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A manual rescaling weight given to each class. If given, it has to</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A manual rescaling weight given to each class. If given, it has to</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  be a 1D Tensor assigning weight to each of the classes. Otherwise,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  be a 1D Tensor assigning weight to each of the classes. Otherwise,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  it is treated as if having all ones.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  it is treated as if having all ones.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2 outputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2 outputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **output** (heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Weighted loss float Tensor. If reduction is 'none', this has the</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Weighted loss float Tensor. If reduction is 'none', this has the</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  shape of [batch_size], or [batch_size, D1, D2, ..., Dk] in case of</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  shape of [batch_size], or [batch_size, D1, D2, ..., Dk] in case of</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  K-dimensional loss. Otherwise, it is a scalar.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  K-dimensional loss. Otherwise, it is a scalar.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **log_prob** (optional, heterogeneous) - **T**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **log_prob** (optional, heterogeneous) - **T**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Log probability tensor. If the output of softmax is prob, its value</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Log probability tensor. If the output of softmax is prob, its value</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  is log(prob).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  is log(prob).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **T** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">77</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(bfloat16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain input and output types to float tensors.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Tind** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **Tind** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain target to integer types</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Constrain target to integer types</code></td></tr>
    </table>

.. _l-onnx-op-softmaxcrossentropyloss-12:

SoftmaxCrossEntropyLoss - 12
============================

**Version**

* **name**: `SoftmaxCrossEntropyLoss (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#SoftmaxCrossEntropyLoss>`_
* **domain**: **main**
* **since_version**: **12**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 12**.

**Summary**

Loss function that measures the softmax cross entropy
between 'scores' and 'labels'.
This operator first computes a loss tensor whose shape is identical to the labels input.
If the input is 2-D with shape (N, C), the loss tensor may be a N-element vector L = (l_1, l_2, ..., l_N).
If the input is N-D tensor with shape (N, C, D1, D2, ..., Dk),
the loss tensor L may have (N, D1, D2, ..., Dk) as its shape and L[i,][j_1][j_2]...[j_k] denotes a scalar element in L.
After L is available, this operator can optionally do a reduction operator.

shape(scores): (N, C) where C is the number of classes, or (N, C, D1, D2,..., Dk),
        with K >= 1 in case of K-dimensional loss.
shape(labels): (N) where each value is 0 <= labels[i] <= C-1, or (N, D1, D2,..., Dk),
        with K >= 1 in case of K-dimensional loss.

The loss for one sample, l_i, can caculated as follows:
    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk], where i is the index of classes.
or
    l[i][d1][d2]...[dk] = -y[i][c][d1][d2]..[dk] * weights[c], if 'weights' is provided.

loss is zero for the case when label-value equals ignore_index.
    l[i][d1][d2]...[dk]  = 0, when labels[n][d1][d2]...[dk] = ignore_index

where:
    p = Softmax(scores)
    y = Log(p)
    c = labels[i][d1][d2]...[dk]

Finally, L is optionally reduced:
If reduction = 'none', the output is L with shape (N, D1, D2, ..., Dk).
If reduction = 'sum', the output is scalar: Sum(L).
If reduction = 'mean', the output is scalar: ReduceMean(L), or if weight is provided: ReduceSum(L) / ReduceSum(W),
where tensor W is of shape (N, D1, D2, ..., Dk) and W[n][d1][d2]...[dk] = weights[labels[i][d1][d2]...[dk]].

**Attributes**

* **ignore_index**:
  Specifies a target value that is ignored and does not contribute to
  the input gradient. It's an optional value.
* **reduction**:
  Type of reduction to apply to loss: none, sum, mean(default).
  'none': no reduction will be applied, 'sum': the output will be
  summed. 'mean': the sum of the output will be divided by the number
  of elements in the output. Default value is ``'mean'``.

**Inputs**

Between 2 and 3 inputs.

* **scores** (heterogeneous) - **T**:
  The predicted outputs with shape [batch_size, class_size], or
  [batch_size, class_size, D1, D2 , ..., Dk], where K is the number of
  dimensions.
* **labels** (heterogeneous) - **Tind**:
  The ground truth output tensor, with shape [batch_size], or
  [batch_size, D1, D2, ..., Dk], where K is the number of dimensions.
  Labels element value shall be in range of [0, C). If ignore_index is
  specified, it may have a value outside [0, C) and the label values
  should either be in the range [0, C) or have the value ignore_index.
* **weights** (optional, heterogeneous) - **T**:
  A manual rescaling weight given to each class. If given, it has to
  be a 1D Tensor assigning weight to each of the classes. Otherwise,
  it is treated as if having all ones.

**Outputs**

Between 1 and 2 outputs.

* **output** (heterogeneous) - **T**:
  Weighted loss float Tensor. If reduction is 'none', this has the
  shape of [batch_size], or [batch_size, D1, D2, ..., Dk] in case of
  K-dimensional loss. Otherwise, it is a scalar.
* **log_prob** (optional, heterogeneous) - **T**:
  Log probability tensor. If the output of softmax is prob, its value
  is log(prob).

**Type Constraints**

* **T** in (
  tensor(double),
  tensor(float),
  tensor(float16)
  ):
  Constrain input and output types to float tensors.
* **Tind** in (
  tensor(int32),
  tensor(int64)
  ):
  Constrain target to integer types
