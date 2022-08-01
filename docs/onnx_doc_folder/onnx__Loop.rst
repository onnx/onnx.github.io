
.. _l-onnx-doc-Loop:

====
Loop
====

.. contents::
    :local:


.. _l-onnx-op-loop-16:

Loop - 16
=========

**Version**

* **name**: `Loop (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Loop>`_
* **domain**: **main**
* **since_version**: **16**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 16**.

**Summary**

Generic Looping construct. This loop has multiple termination conditions:

1) Trip count. Iteration count specified at runtime. Set by
   specifying the input M. Optional. Set to empty string to omit.
   Note that a static trip count (specified at graph construction time) can be
   specified by passing in a constant node for input M.
2) Loop termination condition. This is an input to the op that determines
   whether to run the first iteration and also a loop-carried dependency for
   the body graph. The body graph must yield a value for the condition variable,
   whether this input is provided or not.

This table summarizes the operating modes of this operator with equivalent
C-style code:

    Operator inputs defined as (max_trip_count, condition_var).

    input ("", ""):
        for (int i=0; ; ++i) {
          cond = ... // Note this value is ignored, but is required in the body
        }

    input ("", cond) // Note this is analogous to a while loop
        bool cond = ...;
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input ("", 1) // Note this is analogous to a do-while loop
        bool cond = true
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input (trip_count, "") // Note this is analogous to a for loop
        int trip_count = ...
        for (int i=0; i < trip_count; ++i) {
          cond = ...; // ignored
        }

    input (trip_count, cond)
        int trip_count = ...;
        bool cond = ...;
        for (int i=0; i < trip_count && cond; ++i) {
          cond = ...;
        }

*Sample usage - cond as well as trip count*

    graph predict-net {
      %a = Constant[value = <Scalar Tensor [3]>]()
      %b = Constant[value = <Scalar Tensor [6]>]()
      %keepgoing = Constant[value = <Scalar Tensor [1]>]()
      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()
      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)
      return
    }

    graph body-net (
      %i[INT32, scalar]           // iteration number
      %keepgoing_in[BOOL, scalar] // incoming loop-termination-condition; not used
      %b_in[INT32, scalar]        // incoming value of loop-carried-dependency b
    ) {
      %my_local = Add(%a, %b_in)
      %b_out = Sub(%a, %b_in) // outgoing value of loop-carried-dependency b
      %keepgoing_out = Greater(%my_local, %b_out) // outgoing loop-termination-condition
      %user_defined_val = Add(%b_in, %b_in) // scan-output value to be accumulated
      return %keepgoing_out, %b_out, %user_defined_val
    }

*Sample equivalent C code*

    {
      /* User-defined code (enclosing scope) */
      int a = 3, b = 6;
      bool keepgoing = true; // Analogous to input cond
      /* End user-defined code */

      /* Implicitly-defined code */
      const int max_trip_count = 10; // Analogous to input M
      int user_defined_vals[]; // Imagine this is resizable
      /* End implicitly-defined code */
      /* initialize loop-carried variables and scan-output variables */
      bool keepgoing_out = keepgoing
      int b_out = b

      for (int i=0; i < max_trip_count && keepgoing_out; ++i) {
        /* Implicitly-defined code: bind actual parameter values
           to formal parameter variables of loop-body */
        bool keepgoing_in = keepgoing_out;
        bool b_in = b_out;

        /* User-defined code (loop body) */
        int my_local = a + b_in; // Reading value "a" from the enclosing scope is fine
        b_out = a - b_in;
        keepgoing_out = my_local > b_out;
        user_defined_val = b_in + b_in; // b_in and b_out are different variables
        /* End user-defined code */

        /* Implicitly defined-code */
        user_defined_vals[i] = user_defined_val // accumulate scan-output values
      }
      // int t = my_local; // Can't do this. my_local is not accessible here.

      // The values below are bound to the output variables of the loop and therefore accessible
      // b_out; user_defined_vals; keepgoing_out;
    }

There are several things of note in this code snippet:

1) Values from the enclosing scope (i.e. variable "a" here) are in scope and can
   be referenced in the inputs of the loop.
2) Any values computed in the loop body that needs to be used in a subsequent
   iteration or after the loop are modelled using a pair of variables in the loop-body,
   consisting of an input variable (eg., b_in) and an output variable (eg., b_out).
   These are referred to as loop-carried dependences. The loop operation node
   supplies the input value of the input variable for the first iteration, and
   returns the output value of the output variable produced by the final
   iteration.
3) Scan_output variables are used to implicitly concatenate values computed across
   all the iterations. In the above example, the value of user_defined_val computed
   over all iterations are concatenated and returned as the value of user_defined_vals
   after the loop.
4) Values created in the body cannot be accessed in the enclosing scope,
   except using the mechanism described above.

Note that the semantics of this op support "diagonal" or "wavefront" execution.
(See Step 3 here for an example:
https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).
Frontends should emit multi-layer RNNs as a series of While operators (with
time being the inner looping dimension), with each successive layer consuming
the scan_outputs from the previous layer, possibly going through several
point-wise operators (e.g. dropout, residual connections, linear layer).

The input/output of subgraph (produced by loop node) matching is based on order instead of name. The implementation will figure out the names based on this order.

**Attributes**

* **body** (required):
  The graph run each iteration. It has 2+N inputs: (iteration_num,
  condition, loop carried dependencies...). It has 1+N+K outputs:
  (condition, loop carried dependencies..., scan_outputs...). Each
  scan_output is created by concatenating the value of the specified
  output value at the end of each iteration of the loop. It is an
  error if the dimensions or data type of these scan_outputs change
  across loop iterations.

**Inputs**

Between 2 and 2147483647 inputs.

* **M** (optional, heterogeneous) - **I**:
  A maximum trip-count for the loop specified at runtime. Optional.
  Pass empty string to skip.
* **cond** (optional, heterogeneous) - **B**:
  A boolean termination condition. Optional. Pass empty string to
  skip.
* **v_initial** (variadic) - **V**:
  The initial values of any loop-carried dependencies (values that
  change across loop iterations)

**Outputs**

Between 1 and 2147483647 outputs.

* **v_final_and_scan_outputs** (variadic) - **V**:
  Final N loop carried dependency values then K scan_outputs. Scan
  outputs must be Tensors.

**Type Constraints**

* **V** in (
  optional(seq(tensor(bfloat16))),
  optional(seq(tensor(bool))),
  optional(seq(tensor(complex128))),
  optional(seq(tensor(complex64))),
  optional(seq(tensor(double))),
  optional(seq(tensor(float))),
  optional(seq(tensor(float16))),
  optional(seq(tensor(int16))),
  optional(seq(tensor(int32))),
  optional(seq(tensor(int64))),
  optional(seq(tensor(int8))),
  optional(seq(tensor(string))),
  optional(seq(tensor(uint16))),
  optional(seq(tensor(uint32))),
  optional(seq(tensor(uint64))),
  optional(seq(tensor(uint8))),
  optional(tensor(bfloat16)),
  optional(tensor(bool)),
  optional(tensor(complex128)),
  optional(tensor(complex64)),
  optional(tensor(double)),
  optional(tensor(float)),
  optional(tensor(float16)),
  optional(tensor(int16)),
  optional(tensor(int32)),
  optional(tensor(int64)),
  optional(tensor(int8)),
  optional(tensor(string)),
  optional(tensor(uint16)),
  optional(tensor(uint32)),
  optional(tensor(uint64)),
  optional(tensor(uint8)),
  seq(tensor(bfloat16)),
  seq(tensor(bool)),
  seq(tensor(complex128)),
  seq(tensor(complex64)),
  seq(tensor(double)),
  seq(tensor(float)),
  seq(tensor(float16)),
  seq(tensor(int16)),
  seq(tensor(int32)),
  seq(tensor(int64)),
  seq(tensor(int8)),
  seq(tensor(string)),
  seq(tensor(uint16)),
  seq(tensor(uint32)),
  seq(tensor(uint64)),
  seq(tensor(uint8)),
  tensor(bfloat16),
  tensor(bool),
  tensor(complex128),
  tensor(complex64),
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int16),
  tensor(int32),
  tensor(int64),
  tensor(int8),
  tensor(string),
  tensor(uint16),
  tensor(uint32),
  tensor(uint64),
  tensor(uint8)
  ):
  All Tensor, Sequence(Tensor), Optional(Tensor), and
  Optional(Sequence(Tensor)) types
* **I** in (
  tensor(int64)
  ):
  tensor of int64, which should be a scalar.
* **B** in (
  tensor(bool)
  ):
  tensor of bool, which should be a scalar.

**Examples**

**loop_11**

::

    # Given a tensor x of values [x1, ..., xN], and initial tensor y
    # sum up its elements using a scan
    # returning the final state (y+x1+x2+...+xN) as well the scan_output
    # [y+x1, y+x1+x2, ..., y+x1+x2+...+xN]

    y_in = onnx.helper.make_tensor_value_info('y_in', onnx.TensorProto.FLOAT, [1])
    y_out = onnx.helper.make_tensor_value_info('y_out', onnx.TensorProto.FLOAT, [1])
    scan_out = onnx.helper.make_tensor_value_info('scan_out', onnx.TensorProto.FLOAT, [1])
    cond_in = onnx.helper.make_tensor_value_info('cond_in', onnx.TensorProto.BOOL, [])
    cond_out = onnx.helper.make_tensor_value_info('cond_out', onnx.TensorProto.BOOL, [])
    iter_count = onnx.helper.make_tensor_value_info('iter_count', onnx.TensorProto.INT64, [])

    x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    y = np.array([-2]).astype(np.float32)

    x_const_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['x'],
        value=onnx.helper.make_tensor(
            name='const_tensor_x',
            data_type=onnx.TensorProto.FLOAT,
            dims=x.shape,
            vals=x.flatten().astype(float),
        )
    )

    one_const_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['one'],
        value=onnx.helper.make_tensor(
            name='const_tensor_one',
            data_type=onnx.TensorProto.INT64,
            dims=(),
            vals=[1]
        )
    )

    i_add_node = onnx.helper.make_node(
        'Add',
        inputs=['iter_count', 'one'],
        outputs=['end']
    )

    start_unsqueeze_node = onnx.helper.make_node(
        'Unsqueeze',
        inputs=['iter_count'],
        outputs=['slice_start'],
        axes=[0]
    )

    end_unsqueeze_node = onnx.helper.make_node(
        'Unsqueeze',
        inputs=['end'],
        outputs=['slice_end'],
        axes=[0]
    )

    slice_node = onnx.helper.make_node(
        'Slice',
        inputs=['x', 'slice_start', 'slice_end'],
        outputs=['slice_out']
    )

    y_add_node = onnx.helper.make_node(
        'Add',
        inputs=['y_in', 'slice_out'],
        outputs=['y_out']
    )

    identity_node = onnx.helper.make_node(
        'Identity',
        inputs=['cond_in'],
        outputs=['cond_out']
    )

    scan_identity_node = onnx.helper.make_node(
        'Identity',
        inputs=['y_out'],
        outputs=['scan_out']
    )

    loop_body = onnx.helper.make_graph(
        [identity_node, x_const_node, one_const_node, i_add_node,
         start_unsqueeze_node, end_unsqueeze_node, slice_node, y_add_node,
         scan_identity_node],
        'loop_body',
        [iter_count, cond_in, y_in],
        [cond_out, y_out, scan_out]
    )

    node = onnx.helper.make_node(
        'Loop',
        inputs=['trip_count', 'cond', 'y'],
        outputs=['res_y', 'res_scan'],
        body=loop_body
    )

    trip_count = np.array(5).astype(np.int64)
    res_y = np.array([13]).astype(np.float32)
    cond = np.array(1).astype(bool)
    res_scan = np.array([-1, 1, 4, 8, 13]).astype(np.float32).reshape((5, 1))
    expect(node, inputs=[trip_count, cond, y], outputs=[res_y, res_scan],
           name='test_loop11', opset_imports=[onnx.helper.make_opsetid("", 11)])

**loop_13**

::

    # Given a tensor x of values [x1, ..., xN],
    # Return a sequence of tensors of
    #   [[x1], [x1, x2], ..., [x1, ..., xN]]

    seq_in = onnx.helper.make_tensor_sequence_value_info('seq_in', onnx.TensorProto.FLOAT, None)
    seq_out = onnx.helper.make_tensor_sequence_value_info('seq_out', onnx.TensorProto.FLOAT, None)
    cond_in = onnx.helper.make_tensor_value_info('cond_in', onnx.TensorProto.BOOL, [])
    cond_out = onnx.helper.make_tensor_value_info('cond_out', onnx.TensorProto.BOOL, [])
    iter_count = onnx.helper.make_tensor_value_info('iter_count', onnx.TensorProto.INT64, [])

    x = np.array([1, 2, 3, 4, 5]).astype(np.float32)

    x_const_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['x'],
        value=onnx.helper.make_tensor(
            name='const_tensor_x',
            data_type=onnx.TensorProto.FLOAT,
            dims=x.shape,
            vals=x.flatten().astype(float),
        )
    )

    one_const_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['one'],
        value=onnx.helper.make_tensor(
            name='const_tensor_one',
            data_type=onnx.TensorProto.INT64,
            dims=(),
            vals=[1]
        )
    )

    zero_const_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['slice_start'],
        value=onnx.helper.make_tensor(
            name='const_tensor_zero',
            data_type=onnx.TensorProto.INT64,
            dims=(1,),
            vals=[0]
        )
    )

    axes_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['axes'],
        value=onnx.helper.make_tensor(
            name='const_tensor_axes',
            data_type=onnx.TensorProto.INT64,
            dims=(),
            vals=[0]
        )
    )

    add_node = onnx.helper.make_node(
        'Add',
        inputs=['iter_count', 'one'],
        outputs=['end']
    )

    end_unsqueeze_node = onnx.helper.make_node(
        'Unsqueeze',
        inputs=['end', 'axes'],
        outputs=['slice_end']
    )

    slice_node = onnx.helper.make_node(
        'Slice',
        inputs=['x', 'slice_start', 'slice_end'],
        outputs=['slice_out']
    )

    insert_node = onnx.helper.make_node(
        'SequenceInsert',
        inputs=['seq_in', 'slice_out'],
        outputs=['seq_out']
    )

    identity_node = onnx.helper.make_node(
        'Identity',
        inputs=['cond_in'],
        outputs=['cond_out']
    )

    loop_body = onnx.helper.make_graph(
        [identity_node, x_const_node, one_const_node, zero_const_node, add_node,
         axes_node, end_unsqueeze_node, slice_node, insert_node],
        'loop_body',
        [iter_count, cond_in, seq_in],
        [cond_out, seq_out]
    )

    node = onnx.helper.make_node(
        'Loop',
        inputs=['trip_count', 'cond', 'seq_empty'],
        outputs=['seq_res'],
        body=loop_body
    )

    trip_count = np.array(5).astype(np.int64)
    seq_empty: List[Any] = []
    seq_res = [x[:int(i)] for i in x]
    cond = np.array(1).astype(bool)
    expect(node, inputs=[trip_count, cond, seq_empty], outputs=[seq_res],
           name='test_loop13_seq', opset_imports=[onnx.helper.make_opsetid("", 13)],
           input_type_protos=[onnx.helper.make_tensor_type_proto(onnx.TensorProto.INT64, trip_count.shape),
                              onnx.helper.make_tensor_type_proto(onnx.TensorProto.BOOL, cond.shape),
                              onnx.helper.make_sequence_type_proto(
                                  onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, []))])

**loop_16_none**

::

    # Given a tensor sequence of values [x1, ..., xN], and an initial optional sequence of tensors [x0],
    # Return a concatenated sequence of tensors of
    #   [x0, [x1], [x1, x2], ..., [x1, ..., xN]]

    ten_in_tp = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [])
    seq_in_tp = onnx.helper.make_sequence_type_proto(ten_in_tp)
    opt_in_tp = onnx.helper.make_optional_type_proto(seq_in_tp)
    opt_in = onnx.helper.make_value_info('opt_seq_in', opt_in_tp)
    seq_out = onnx.helper.make_tensor_sequence_value_info('seq_out', onnx.TensorProto.FLOAT, [])
    cond_in = onnx.helper.make_tensor_value_info('cond_in', onnx.TensorProto.BOOL, [])
    cond_out = onnx.helper.make_tensor_value_info('cond_out', onnx.TensorProto.BOOL, [])
    iter_count = onnx.helper.make_tensor_value_info('iter_count', onnx.TensorProto.INT64, [])

    x0 = np.array(0).astype(np.float32)
    x = np.array([1, 2, 3, 4, 5]).astype(np.float32)

    optional_has_elem_node = onnx.helper.make_node(
        'OptionalHasElement',
        inputs=['opt_seq_in'],
        outputs=['optional_has_elem']
    )

    optional_is_none = onnx.helper.make_node(
        'Not',
        inputs=['optional_has_elem'],
        outputs=['optional_is_none']
    )

    optional_get_elem = onnx.helper.make_node(
        'OptionalGetElement',
        inputs=['opt_seq_in'],
        outputs=['seq_in']
    )

    constant_in = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['constant_in'],
        value=onnx.helper.make_tensor(
            name='const_tensor',
            data_type=onnx.TensorProto.FLOAT,
            dims=(),
            vals=[0]
        )
    )

    seq_const_in = onnx.helper.make_node(
        'SequenceConstruct',
        inputs=['constant_in'],
        outputs=['init_seq_in']
    )

    then_seq_out = onnx.helper.make_tensor_sequence_value_info('init_seq_in', onnx.TensorProto.FLOAT, [])
    then_body = onnx.helper.make_graph(
        [constant_in, seq_const_in],
        'then_body',
        [],
        [then_seq_out]
    )

    else_seq_out = onnx.helper.make_tensor_sequence_value_info('seq_in', onnx.TensorProto.FLOAT, [])
    else_body = onnx.helper.make_graph(
        [optional_get_elem],
        'else_body',
        [],
        [else_seq_out]
    )

    if_node = onnx.helper.make_node(
        'If',
        inputs=['optional_is_none'],
        outputs=['sequence'],
        then_branch=then_body,
        else_branch=else_body
    )

    x_const_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['x'],
        value=onnx.helper.make_tensor(
            name='const_tensor_x',
            data_type=onnx.TensorProto.FLOAT,
            dims=x.shape,
            vals=x.flatten().astype(float),
        )
    )

    one_const_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['one'],
        value=onnx.helper.make_tensor(
            name='const_tensor_one',
            data_type=onnx.TensorProto.INT64,
            dims=(),
            vals=[1]
        )
    )

    zero_const_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['slice_start'],
        value=onnx.helper.make_tensor(
            name='const_tensor_zero',
            data_type=onnx.TensorProto.INT64,
            dims=(1,),
            vals=[0]
        )
    )

    axes_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=['axes'],
        value=onnx.helper.make_tensor(
            name='const_tensor_axes',
            data_type=onnx.TensorProto.INT64,
            dims=(),
            vals=[0]
        )
    )

    add_node = onnx.helper.make_node(
        'Add',
        inputs=['iter_count', 'one'],
        outputs=['end']
    )

    end_unsqueeze_node = onnx.helper.make_node(
        'Unsqueeze',
        inputs=['end', 'axes'],
        outputs=['slice_end']
    )

    slice_node = onnx.helper.make_node(
        'Slice',
        inputs=['x', 'slice_start', 'slice_end'],
        outputs=['slice_out']
    )

    insert_node = onnx.helper.make_node(
        'SequenceInsert',
        inputs=['sequence', 'slice_out'],
        outputs=['seq_out']
    )

    identity_node = onnx.helper.make_node(
        'Identity',
        inputs=['cond_in'],
        outputs=['cond_out']
    )

    loop_body = onnx.helper.make_graph(
        [identity_node, optional_has_elem_node, optional_is_none, if_node, x_const_node, one_const_node,
         zero_const_node, add_node, axes_node, end_unsqueeze_node, slice_node, insert_node],
        'loop_body',
        [iter_count, cond_in, opt_in],
        [cond_out, seq_out]
    )

    node = onnx.helper.make_node(
        'Loop',
        inputs=['trip_count', 'cond', 'opt_seq'],
        outputs=['seq_res'],
        body=loop_body
    )

    trip_count = np.array(5).astype(np.int64)
    cond = np.array(1).astype(bool)
    seq_res = compute_loop_outputs(x, [x0], trip_count)
    opt_seq_in: List[Any] = [x0]
    expect(node, inputs=[trip_count, cond, opt_seq_in], outputs=[seq_res],
           name='test_loop16_seq_none', opset_imports=[onnx.helper.make_opsetid("", 16)],
           input_type_protos=[onnx.helper.make_tensor_type_proto(onnx.TensorProto.INT64, trip_count.shape),
                              onnx.helper.make_tensor_type_proto(onnx.TensorProto.BOOL, cond.shape),
                              opt_in_tp])

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Generic Looping construct. This loop has multiple termination conditions:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Generic Looping construct. This loop has multiple termination conditions:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1) Trip count. Iteration count specified at runtime. Set by</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1) Trip count. Iteration count specified at runtime. Set by</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   specifying the input M. Optional. Set to empty string to omit.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   specifying the input M. Optional. Set to empty string to omit.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   Note that a static trip count (specified at graph construction time) can be</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   Note that a static trip count (specified at graph construction time) can be</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   specified by passing in a constant node for input M.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   specified by passing in a constant node for input M.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2) Loop termination condition. This is an input to the op that determines</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2) Loop termination condition. This is an input to the op that determines</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   whether to run the first iteration and also a loop-carried dependency for</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   whether to run the first iteration and also a loop-carried dependency for</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   the body graph. The body graph must yield a value for the condition variable,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   the body graph. The body graph must yield a value for the condition variable,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   whether this input is provided or not.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   whether this input is provided or not.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This table summarizes the operating modes of this operator with equivalent</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This table summarizes the operating modes of this operator with equivalent</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">C-style code:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">C-style code:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    Operator inputs defined as (max_trip_count, condition_var).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    Operator inputs defined as (max_trip_count, condition_var).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input ("", ""):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input ("", ""):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; ; ++i) {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; ; ++i) {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ... // Note this value is ignored, but is required in the body</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ... // Note this value is ignored, but is required in the body</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input ("", cond) // Note this is analogous to a while loop</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input ("", cond) // Note this is analogous to a while loop</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool cond = ...;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool cond = ...;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; cond; ++i) {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; cond; ++i) {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input ("", 1) // Note this is analogous to a do-while loop</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input ("", 1) // Note this is analogous to a do-while loop</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool cond = true</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool cond = true</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; cond; ++i) {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; cond; ++i) {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input (trip_count, "") // Note this is analogous to a for loop</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input (trip_count, "") // Note this is analogous to a for loop</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        int trip_count = ...</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        int trip_count = ...</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; i < trip_count; ++i) {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; i < trip_count; ++i) {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...; // ignored</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...; // ignored</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input (trip_count, cond)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input (trip_count, cond)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        int trip_count = ...;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        int trip_count = ...;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool cond = ...;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool cond = ...;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; i < trip_count && cond; ++i) {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; i < trip_count && cond; ++i) {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">*Sample usage - cond as well as trip count*</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">*Sample usage - cond as well as trip count*</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    graph predict-net {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    graph predict-net {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %a = Constant[value = <Scalar Tensor [3]>]()</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %a = Constant[value = <Scalar Tensor [3]>]()</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %b = Constant[value = <Scalar Tensor [6]>]()</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %b = Constant[value = <Scalar Tensor [6]>]()</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %keepgoing = Constant[value = <Scalar Tensor [1]>]()</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %keepgoing = Constant[value = <Scalar Tensor [1]>]()</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      return</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      return</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    graph body-net (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    graph body-net (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %i[INT32, scalar]           // iteration number</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %i[INT32, scalar]           // iteration number</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %keepgoing_in[BOOL, scalar] // incoming loop-termination-condition; not used</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %keepgoing_in[BOOL, scalar] // incoming loop-termination-condition; not used</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %b_in[INT32, scalar]        // incoming value of loop-carried-dependency b</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %b_in[INT32, scalar]        // incoming value of loop-carried-dependency b</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    ) {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    ) {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %my_local = Add(%a, %b_in)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %my_local = Add(%a, %b_in)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %b_out = Sub(%a, %b_in) // outgoing value of loop-carried-dependency b</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %b_out = Sub(%a, %b_in) // outgoing value of loop-carried-dependency b</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %keepgoing_out = Greater(%my_local, %b_out) // outgoing loop-termination-condition</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %keepgoing_out = Greater(%my_local, %b_out) // outgoing loop-termination-condition</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %user_defined_val = Add(%b_in, %b_in) // scan-output value to be accumulated</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %user_defined_val = Add(%b_in, %b_in) // scan-output value to be accumulated</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      return %keepgoing_out, %b_out, %user_defined_val</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      return %keepgoing_out, %b_out, %user_defined_val</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">*Sample equivalent C code*</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">*Sample equivalent C code*</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* User-defined code (enclosing scope) */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* User-defined code (enclosing scope) */</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      int a = 3, b = 6;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      int a = 3, b = 6;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      bool keepgoing = true; // Analogous to input cond</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      bool keepgoing = true; // Analogous to input cond</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* End user-defined code */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* End user-defined code */</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* Implicitly-defined code */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* Implicitly-defined code */</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      const int max_trip_count = 10; // Analogous to input M</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      const int max_trip_count = 10; // Analogous to input M</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      int user_defined_vals[]; // Imagine this is resizable</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      int user_defined_vals[]; // Imagine this is resizable</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* End implicitly-defined code */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* End implicitly-defined code */</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* initialize loop-carried variables and scan-output variables */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* initialize loop-carried variables and scan-output variables */</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      bool keepgoing_out = keepgoing</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      bool keepgoing_out = keepgoing</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      int b_out = b</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      int b_out = b</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      for (int i=0; i < max_trip_count && keepgoing_out; ++i) {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      for (int i=0; i < max_trip_count && keepgoing_out; ++i) {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        /* Implicitly-defined code: bind actual parameter values</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        /* Implicitly-defined code: bind actual parameter values</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">           to formal parameter variables of loop-body */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">           to formal parameter variables of loop-body */</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool keepgoing_in = keepgoing_out;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool keepgoing_in = keepgoing_out;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool b_in = b_out;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool b_in = b_out;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        /* User-defined code (loop body) */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        /* User-defined code (loop body) */</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        int my_local = a + b_in; // Reading value "a" from the enclosing scope is fine</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        int my_local = a + b_in; // Reading value "a" from the enclosing scope is fine</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        b_out = a - b_in;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        b_out = a - b_in;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        keepgoing_out = my_local > b_out;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        keepgoing_out = my_local > b_out;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        user_defined_val = b_in + b_in; // b_in and b_out are different variables</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        user_defined_val = b_in + b_in; // b_in and b_out are different variables</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        /* End user-defined code */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        /* End user-defined code */</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        /* Implicitly defined-code */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        /* Implicitly defined-code */</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        user_defined_vals[i] = user_defined_val // accumulate scan-output values</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        user_defined_vals[i] = user_defined_val // accumulate scan-output values</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      // int t = my_local; // Can't do this. my_local is not accessible here.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      // int t = my_local; // Can't do this. my_local is not accessible here.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      // The values below are bound to the output variables of the loop and therefore accessible</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      // The values below are bound to the output variables of the loop and therefore accessible</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      // b_out; user_defined_vals; keepgoing_out;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      // b_out; user_defined_vals; keepgoing_out;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">There are several things of note in this code snippet:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">There are several things of note in this code snippet:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">109</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">109</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1) Values from the enclosing scope (i.e. variable "a" here) are in scope and can</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1) Values from the enclosing scope (i.e. variable "a" here) are in scope and can</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">110</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">110</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   be referenced in the inputs of the loop.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   be referenced in the inputs of the loop.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">111</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">111</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2) Any values computed in the loop body that needs to be used in a subsequent</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2) Any values computed in the loop body that needs to be used in a subsequent</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">112</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">112</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   iteration or after the loop are modelled using a pair of variables in the loop-body,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   iteration or after the loop are modelled using a pair of variables in the loop-body,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">113</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">113</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   consisting of an input variable (eg., b_in) and an output variable (eg., b_out).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   consisting of an input variable (eg., b_in) and an output variable (eg., b_out).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">114</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">114</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   These are referred to as loop-carried dependences. The loop operation node</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   These are referred to as loop-carried dependences. The loop operation node</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">115</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">115</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   supplies the input value of the input variable for the first iteration, and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   supplies the input value of the input variable for the first iteration, and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">116</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">116</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   returns the output value of the output variable produced by the final</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   returns the output value of the output variable produced by the final</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">117</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">117</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   iteration.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   iteration.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">118</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">118</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3) Scan_output variables are used to implicitly concatenate values computed across</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3) Scan_output variables are used to implicitly concatenate values computed across</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">119</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">119</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   all the iterations. In the above example, the value of user_defined_val computed</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   all the iterations. In the above example, the value of user_defined_val computed</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">120</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">120</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   over all iterations are concatenated and returned as the value of user_defined_vals</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   over all iterations are concatenated and returned as the value of user_defined_vals</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">121</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">121</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   after the loop.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   after the loop.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">122</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">122</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4) Values created in the body cannot be accessed in the enclosing scope,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4) Values created in the body cannot be accessed in the enclosing scope,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">123</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">123</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   except using the mechanism described above.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   except using the mechanism described above.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">124</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">124</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">125</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">125</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Note that the semantics of this op support "diagonal" or "wavefront" execution.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Note that the semantics of this op support "diagonal" or "wavefront" execution.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">126</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">126</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">(See Step 3 here for an example:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">(See Step 3 here for an example:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">127</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">127</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">128</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">128</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Frontends should emit multi-layer RNNs as a series of While operators (with</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Frontends should emit multi-layer RNNs as a series of While operators (with</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">129</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">129</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">time being the inner looping dimension), with each successive layer consuming</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">time being the inner looping dimension), with each successive layer consuming</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">130</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">130</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the scan_outputs from the previous layer, possibly going through several</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the scan_outputs from the previous layer, possibly going through several</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">131</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">131</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">point-wise operators (e.g. dropout, residual connections, linear layer).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">point-wise operators (e.g. dropout, residual connections, linear layer).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">132</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">132</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">133</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">133</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The input/output of subgraph (produced by loop node) matching is based on order instead of name. The implementation will figure out the names based on this order.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">The input/output of subgraph (produced by loop node) matching is based on order instead of name. The implementation will figure out the names based on this order.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">134</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">134</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">135</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">135</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">136</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">136</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">137</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">137</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **body** (required):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **body** (required):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">138</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">138</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The graph run each iteration. It has 2+N inputs: (iteration_num,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The graph run each iteration. It has 2+N inputs: (iteration_num,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">139</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">139</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  condition, loop carried dependencies...). It has 1+N+K outputs:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  condition, loop carried dependencies...). It has 1+N+K outputs:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">140</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">140</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (condition, loop carried dependencies..., scan_outputs...). Each</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (condition, loop carried dependencies..., scan_outputs...). Each</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">141</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">141</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  scan_output is created by concatenating the value of the specified</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  scan_output is created by concatenating the value of the specified</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">142</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">142</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output value at the end of each iteration of the loop. It is an</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output value at the end of each iteration of the loop. It is an</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">143</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">143</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  error if the dimensions or data type of these scan_outputs change</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  error if the dimensions or data type of these scan_outputs change</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">144</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">144</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  across loop iterations.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  across loop iterations.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">145</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">145</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">146</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">146</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">147</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">147</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">148</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">148</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 2147483647 inputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 2147483647 inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">149</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">149</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">150</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">150</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **M** (optional, heterogeneous) - **I**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **M** (optional, heterogeneous) - **I**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">151</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">151</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A maximum trip-count for the loop specified at runtime. Optional.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A maximum trip-count for the loop specified at runtime. Optional.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">152</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">152</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Pass empty string to skip.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Pass empty string to skip.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">153</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">153</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **cond** (optional, heterogeneous) - **B**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **cond** (optional, heterogeneous) - **B**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">154</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">154</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A boolean termination condition. Optional. Pass empty string to</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A boolean termination condition. Optional. Pass empty string to</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">155</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">155</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  skip.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  skip.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">156</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">156</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **v_initial** (variadic) - **V**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **v_initial** (variadic) - **V**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">157</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">157</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The initial values of any loop-carried dependencies (values that</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The initial values of any loop-carried dependencies (values that</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">158</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">158</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  change across loop iterations)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  change across loop iterations)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">159</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">159</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">160</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">160</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">161</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">161</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">162</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">162</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2147483647 outputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2147483647 outputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">163</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">163</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">164</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">164</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **v_final_and_scan_outputs** (variadic) - **V**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **v_final_and_scan_outputs** (variadic) - **V**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">165</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">165</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Final N loop carried dependency values then K scan_outputs. Scan</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Final N loop carried dependency values then K scan_outputs. Scan</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">166</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">166</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  outputs must be Tensors.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  outputs must be Tensors.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">167</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">167</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">168</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">168</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">169</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">169</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">170</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">170</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **V** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **V** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">171</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(seq(tensor(bfloat16))),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">172</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(seq(tensor(bool))),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">173</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(seq(tensor(complex128))),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">174</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(seq(tensor(complex64))),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">175</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(seq(tensor(double))),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">176</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(seq(tensor(float))),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">177</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(seq(tensor(float16))),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">178</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(seq(tensor(int16))),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">179</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(seq(tensor(int32))),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">180</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(seq(tensor(int64))),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">181</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(seq(tensor(int8))),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">182</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(seq(tensor(string))),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">183</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(seq(tensor(uint16))),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">184</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(seq(tensor(uint32))),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">185</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(seq(tensor(uint64))),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">186</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(seq(tensor(uint8))),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">187</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(tensor(bfloat16)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">188</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(tensor(bool)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">189</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(tensor(complex128)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">190</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(tensor(complex64)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">191</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(tensor(double)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">192</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(tensor(float)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">193</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(tensor(float16)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">194</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(tensor(int16)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">195</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(tensor(int32)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">196</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(tensor(int64)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">197</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(tensor(int8)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">198</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(tensor(string)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">199</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(tensor(uint16)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">200</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(tensor(uint32)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">201</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(tensor(uint64)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">202</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  optional(tensor(uint8)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">203</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  seq(tensor(bfloat16)),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">171</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">204</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(bool)),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(bool)),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">172</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">205</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(complex128)),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(complex128)),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">173</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">206</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(complex64)),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(complex64)),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">174</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">207</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(double)),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(double)),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">175</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">208</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(float)),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(float)),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">176</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">209</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(float16)),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(float16)),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">177</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">210</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(int16)),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(int16)),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">178</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">211</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(int32)),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(int32)),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">179</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">212</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(int64)),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(int64)),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">180</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">213</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(int8)),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(int8)),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">181</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">214</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(string)),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(string)),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">182</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">215</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(uint16)),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(uint16)),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">183</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">216</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(uint32)),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(uint32)),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">184</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">217</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(uint64)),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(uint64)),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">185</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">218</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(uint8)),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  seq(tensor(uint8)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">219</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  tensor(bfloat16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">186</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">220</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">187</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">221</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">188</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">222</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">189</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">223</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">190</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">224</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">191</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">225</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">192</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">226</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">193</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">227</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">194</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">228</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">195</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">229</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">196</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">230</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">197</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">231</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">198</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">232</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">199</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">233</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">200</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">234</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">201</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">235</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td><code>202</code></td><td><code>236</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  All Tensor <span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;"> </span>Sequence t<span style="color:#BA4A00;">y</span><span style="color:#BA4A00;">p</span>es</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  All Tensor<span style="color:#196F3D;">,</span> Sequence<span style="color:#196F3D;">(</span><span style="color:#196F3D;">T</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;">,</span> <span style="color:#196F3D;">O</span><span style="color:#196F3D;">p</span>t<span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">(</span><span style="color:#196F3D;">T</span>e<span style="color:#196F3D;">n</span>s<span style="color:#196F3D;">o</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">)</span><span style="color:#196F3D;">,</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">237</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  Optional(Sequence(Tensor)) types</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">203</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">238</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **I** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **I** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">204</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">239</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">205</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">240</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">206</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">241</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor of int64, which should be a scalar.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor of int64, which should be a scalar.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">207</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">242</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">208</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">243</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">209</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">244</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">210</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">245</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor of bool, which should be a scalar.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor of bool, which should be a scalar.</code></td></tr>
    </table>

.. _l-onnx-op-loop-13:

Loop - 13
=========

**Version**

* **name**: `Loop (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Loop>`_
* **domain**: **main**
* **since_version**: **13**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 13**.

**Summary**

Generic Looping construct. This loop has multiple termination conditions:

1) Trip count. Iteration count specified at runtime. Set by
   specifying the input M. Optional. Set to empty string to omit.
   Note that a static trip count (specified at graph construction time) can be
   specified by passing in a constant node for input M.
2) Loop termination condition. This is an input to the op that determines
   whether to run the first iteration and also a loop-carried dependency for
   the body graph. The body graph must yield a value for the condition variable,
   whether this input is provided or not.

This table summarizes the operating modes of this operator with equivalent
C-style code:

    Operator inputs defined as (max_trip_count, condition_var).

    input ("", ""):
        for (int i=0; ; ++i) {
          cond = ... // Note this value is ignored, but is required in the body
        }

    input ("", cond) // Note this is analogous to a while loop
        bool cond = ...;
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input ("", 1) // Note this is analogous to a do-while loop
        bool cond = true
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input (trip_count, "") // Note this is analogous to a for loop
        int trip_count = ...
        for (int i=0; i < trip_count; ++i) {
          cond = ...; // ignored
        }

    input (trip_count, cond)
        int trip_count = ...;
        bool cond = ...;
        for (int i=0; i < trip_count && cond; ++i) {
          cond = ...;
        }

*Sample usage - cond as well as trip count*

    graph predict-net {
      %a = Constant[value = <Scalar Tensor [3]>]()
      %b = Constant[value = <Scalar Tensor [6]>]()
      %keepgoing = Constant[value = <Scalar Tensor [1]>]()
      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()
      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)
      return
    }

    graph body-net (
      %i[INT32, scalar]           // iteration number
      %keepgoing_in[BOOL, scalar] // incoming loop-termination-condition; not used
      %b_in[INT32, scalar]        // incoming value of loop-carried-dependency b
    ) {
      %my_local = Add(%a, %b_in)
      %b_out = Sub(%a, %b_in) // outgoing value of loop-carried-dependency b
      %keepgoing_out = Greater(%my_local, %b_out) // outgoing loop-termination-condition
      %user_defined_val = Add(%b_in, %b_in) // scan-output value to be accumulated
      return %keepgoing_out, %b_out, %user_defined_val
    }

*Sample equivalent C code*

    {
      /* User-defined code (enclosing scope) */
      int a = 3, b = 6;
      bool keepgoing = true; // Analogous to input cond
      /* End user-defined code */

      /* Implicitly-defined code */
      const int max_trip_count = 10; // Analogous to input M
      int user_defined_vals[]; // Imagine this is resizable
      /* End implicitly-defined code */
      /* initialize loop-carried variables and scan-output variables */
      bool keepgoing_out = keepgoing
      int b_out = b

      for (int i=0; i < max_trip_count && keepgoing_out; ++i) {
        /* Implicitly-defined code: bind actual parameter values
           to formal parameter variables of loop-body */
        bool keepgoing_in = keepgoing_out;
        bool b_in = b_out;

        /* User-defined code (loop body) */
        int my_local = a + b_in; // Reading value "a" from the enclosing scope is fine
        b_out = a - b_in;
        keepgoing_out = my_local > b_out;
        user_defined_val = b_in + b_in; // b_in and b_out are different variables
        /* End user-defined code */

        /* Implicitly defined-code */
        user_defined_vals[i] = user_defined_val // accumulate scan-output values
      }
      // int t = my_local; // Can't do this. my_local is not accessible here.

      // The values below are bound to the output variables of the loop and therefore accessible
      // b_out; user_defined_vals; keepgoing_out;
    }

There are several things of note in this code snippet:

1) Values from the enclosing scope (i.e. variable "a" here) are in scope and can
   be referenced in the inputs of the loop.
2) Any values computed in the loop body that needs to be used in a subsequent
   iteration or after the loop are modelled using a pair of variables in the loop-body,
   consisting of an input variable (eg., b_in) and an output variable (eg., b_out).
   These are referred to as loop-carried dependences. The loop operation node
   supplies the input value of the input variable for the first iteration, and
   returns the output value of the output variable produced by the final
   iteration.
3) Scan_output variables are used to implicitly concatenate values computed across
   all the iterations. In the above example, the value of user_defined_val computed
   over all iterations are concatenated and returned as the value of user_defined_vals
   after the loop.
4) Values created in the body cannot be accessed in the enclosing scope,
   except using the mechanism described above.

Note that the semantics of this op support "diagonal" or "wavefront" execution.
(See Step 3 here for an example:
https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).
Frontends should emit multi-layer RNNs as a series of While operators (with
time being the inner looping dimension), with each successive layer consuming
the scan_outputs from the previous layer, possibly going through several
point-wise operators (e.g. dropout, residual connections, linear layer).

The input/output of subgraph (produced by loop node) matching is based on order instead of name. The implementation will figure out the names based on this order.

**Attributes**

* **body** (required):
  The graph run each iteration. It has 2+N inputs: (iteration_num,
  condition, loop carried dependencies...). It has 1+N+K outputs:
  (condition, loop carried dependencies..., scan_outputs...). Each
  scan_output is created by concatenating the value of the specified
  output value at the end of each iteration of the loop. It is an
  error if the dimensions or data type of these scan_outputs change
  across loop iterations.

**Inputs**

Between 2 and 2147483647 inputs.

* **M** (optional, heterogeneous) - **I**:
  A maximum trip-count for the loop specified at runtime. Optional.
  Pass empty string to skip.
* **cond** (optional, heterogeneous) - **B**:
  A boolean termination condition. Optional. Pass empty string to
  skip.
* **v_initial** (variadic) - **V**:
  The initial values of any loop-carried dependencies (values that
  change across loop iterations)

**Outputs**

Between 1 and 2147483647 outputs.

* **v_final_and_scan_outputs** (variadic) - **V**:
  Final N loop carried dependency values then K scan_outputs. Scan
  outputs must be Tensors.

**Type Constraints**

* **V** in (
  seq(tensor(bool)),
  seq(tensor(complex128)),
  seq(tensor(complex64)),
  seq(tensor(double)),
  seq(tensor(float)),
  seq(tensor(float16)),
  seq(tensor(int16)),
  seq(tensor(int32)),
  seq(tensor(int64)),
  seq(tensor(int8)),
  seq(tensor(string)),
  seq(tensor(uint16)),
  seq(tensor(uint32)),
  seq(tensor(uint64)),
  seq(tensor(uint8)),
  tensor(bool),
  tensor(complex128),
  tensor(complex64),
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int16),
  tensor(int32),
  tensor(int64),
  tensor(int8),
  tensor(string),
  tensor(uint16),
  tensor(uint32),
  tensor(uint64),
  tensor(uint8)
  ):
  All Tensor and Sequence types
* **I** in (
  tensor(int64)
  ):
  tensor of int64, which should be a scalar.
* **B** in (
  tensor(bool)
  ):
  tensor of bool, which should be a scalar.

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Generic Looping construct. This loop has multiple termination conditions:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Generic Looping construct. This loop has multiple termination conditions:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1) Trip count. Iteration count specified at runtime. Set by</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1) Trip count. Iteration count specified at runtime. Set by</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   specifying the input M. Optional. Set to empty string to omit.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   specifying the input M. Optional. Set to empty string to omit.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   Note that a static trip count (specified at graph construction time) can be</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   Note that a static trip count (specified at graph construction time) can be</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   specified by passing in a constant node for input M.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   specified by passing in a constant node for input M.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2) Loop termination condition. This is an input to the op that determines</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2) Loop termination condition. This is an input to the op that determines</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   whether to run the first iteration and also a loop-carried dependency for</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   whether to run the first iteration and also a loop-carried dependency for</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   the body graph. The body graph must yield a value for the condition variable,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   the body graph. The body graph must yield a value for the condition variable,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   whether this input is provided or not.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   whether this input is provided or not.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This table summarizes the operating modes of this operator with equivalent</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This table summarizes the operating modes of this operator with equivalent</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">C-style code:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">C-style code:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    Operator inputs defined as (max_trip_count, condition_var).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    Operator inputs defined as (max_trip_count, condition_var).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input ("", ""):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input ("", ""):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; ; ++i) {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; ; ++i) {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ... // Note this value is ignored, but is required in the body</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ... // Note this value is ignored, but is required in the body</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input ("", cond) // Note this is analogous to a while loop</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input ("", cond) // Note this is analogous to a while loop</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool cond = ...;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool cond = ...;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; cond; ++i) {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; cond; ++i) {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input ("", 1) // Note this is analogous to a do-while loop</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input ("", 1) // Note this is analogous to a do-while loop</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool cond = true</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool cond = true</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; cond; ++i) {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; cond; ++i) {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input (trip_count, "") // Note this is analogous to a for loop</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input (trip_count, "") // Note this is analogous to a for loop</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        int trip_count = ...</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        int trip_count = ...</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; i < trip_count; ++i) {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; i < trip_count; ++i) {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...; // ignored</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...; // ignored</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input (trip_count, cond)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input (trip_count, cond)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        int trip_count = ...;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        int trip_count = ...;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool cond = ...;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool cond = ...;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; i < trip_count && cond; ++i) {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; i < trip_count && cond; ++i) {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">*Sample usage - cond as well as trip count*</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">*Sample usage - cond as well as trip count*</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    graph predict-net {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    graph predict-net {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %a = Constant[value = <Scalar Tensor [3]>]()</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %a = Constant[value = <Scalar Tensor [3]>]()</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %b = Constant[value = <Scalar Tensor [6]>]()</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %b = Constant[value = <Scalar Tensor [6]>]()</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %keepgoing = Constant[value = <Scalar Tensor [1]>]()</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %keepgoing = Constant[value = <Scalar Tensor [1]>]()</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      return</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      return</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    graph body-net (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    graph body-net (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">58</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %i[INT32, scalar]           // iteration number</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %i[INT32, scalar]           // iteration number</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">59</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %keepgoing_in[BOOL, scalar] // incoming loop-termination-condition; not used</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %keepgoing_in[BOOL, scalar] // incoming loop-termination-condition; not used</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">60</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %b_in[INT32, scalar]        // incoming value of loop-carried-dependency b</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %b_in[INT32, scalar]        // incoming value of loop-carried-dependency b</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    ) {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    ) {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">62</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %my_local = Add(%a, %b_in)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %my_local = Add(%a, %b_in)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">63</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %b_out = Sub(%a, %b_in) // outgoing value of loop-carried-dependency b</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %b_out = Sub(%a, %b_in) // outgoing value of loop-carried-dependency b</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">64</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %keepgoing_out = Greater(%my_local, %b_out) // outgoing loop-termination-condition</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %keepgoing_out = Greater(%my_local, %b_out) // outgoing loop-termination-condition</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">65</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %user_defined_val = Add(%b_in, %b_in) // scan-output value to be accumulated</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %user_defined_val = Add(%b_in, %b_in) // scan-output value to be accumulated</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">66</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      return %keepgoing_out, %b_out, %user_defined_val</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      return %keepgoing_out, %b_out, %user_defined_val</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">*Sample equivalent C code*</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">*Sample equivalent C code*</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* User-defined code (enclosing scope) */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* User-defined code (enclosing scope) */</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      int a = 3, b = 6;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      int a = 3, b = 6;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      bool keepgoing = true; // Analogous to input cond</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      bool keepgoing = true; // Analogous to input cond</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* End user-defined code */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* End user-defined code */</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* Implicitly-defined code */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* Implicitly-defined code */</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      const int max_trip_count = 10; // Analogous to input M</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      const int max_trip_count = 10; // Analogous to input M</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      int user_defined_vals[]; // Imagine this is resizable</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      int user_defined_vals[]; // Imagine this is resizable</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* End implicitly-defined code */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* End implicitly-defined code */</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">81</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* initialize loop-carried variables and scan-output variables */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* initialize loop-carried variables and scan-output variables */</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      bool keepgoing_out = keepgoing</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      bool keepgoing_out = keepgoing</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">83</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      int b_out = b</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      int b_out = b</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">84</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">85</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      for (int i=0; i < max_trip_count && keepgoing_out; ++i) {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      for (int i=0; i < max_trip_count && keepgoing_out; ++i) {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">86</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        /* Implicitly-defined code: bind actual parameter values</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        /* Implicitly-defined code: bind actual parameter values</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">87</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">           to formal parameter variables of loop-body */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">           to formal parameter variables of loop-body */</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">88</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool keepgoing_in = keepgoing_out;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool keepgoing_in = keepgoing_out;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">89</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool b_in = b_out;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool b_in = b_out;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        /* User-defined code (loop body) */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        /* User-defined code (loop body) */</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">92</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        int my_local = a + b_in; // Reading value "a" from the enclosing scope is fine</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        int my_local = a + b_in; // Reading value "a" from the enclosing scope is fine</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        b_out = a - b_in;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        b_out = a - b_in;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        keepgoing_out = my_local > b_out;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        keepgoing_out = my_local > b_out;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        user_defined_val = b_in + b_in; // b_in and b_out are different variables</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        user_defined_val = b_in + b_in; // b_in and b_out are different variables</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        /* End user-defined code */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        /* End user-defined code */</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">97</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        /* Implicitly defined-code */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        /* Implicitly defined-code */</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">99</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        user_defined_vals[i] = user_defined_val // accumulate scan-output values</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        user_defined_vals[i] = user_defined_val // accumulate scan-output values</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">100</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">101</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      // int t = my_local; // Can't do this. my_local is not accessible here.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      // int t = my_local; // Can't do this. my_local is not accessible here.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">103</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      // The values below are bound to the output variables of the loop and therefore accessible</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      // The values below are bound to the output variables of the loop and therefore accessible</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      // b_out; user_defined_vals; keepgoing_out;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      // b_out; user_defined_vals; keepgoing_out;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">There are several things of note in this code snippet:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">There are several things of note in this code snippet:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">109</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">109</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1) Values from the enclosing scope (i.e. variable "a" here) are in scope and can</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1) Values from the enclosing scope (i.e. variable "a" here) are in scope and can</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">110</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">110</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   be referenced in the inputs of the loop.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   be referenced in the inputs of the loop.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">111</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">111</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2) Any values computed in the loop body that needs to be used in a subsequent</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2) Any values computed in the loop body that needs to be used in a subsequent</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">112</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">112</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   iteration or after the loop are modelled using a pair of variables in the loop-body,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   iteration or after the loop are modelled using a pair of variables in the loop-body,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">113</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">113</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   consisting of an input variable (eg., b_in) and an output variable (eg., b_out).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   consisting of an input variable (eg., b_in) and an output variable (eg., b_out).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">114</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">114</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   These are referred to as loop-carried dependences. The loop operation node</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   These are referred to as loop-carried dependences. The loop operation node</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">115</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">115</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   supplies the input value of the input variable for the first iteration, and</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   supplies the input value of the input variable for the first iteration, and</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">116</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">116</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   returns the output value of the output variable produced by the final</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   returns the output value of the output variable produced by the final</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">117</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">117</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   iteration.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   iteration.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">118</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">118</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3) Scan_output variables are used to implicitly concatenate values computed across</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3) Scan_output variables are used to implicitly concatenate values computed across</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">119</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">119</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   all the iterations. In the above example, the value of user_defined_val computed</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   all the iterations. In the above example, the value of user_defined_val computed</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">120</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">120</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   over all iterations are concatenated and returned as the value of user_defined_vals</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   over all iterations are concatenated and returned as the value of user_defined_vals</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">121</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">121</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   after the loop.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   after the loop.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">122</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">122</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4) Values created in the body cannot be accessed in the enclosing scope,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4) Values created in the body cannot be accessed in the enclosing scope,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">123</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">123</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   except using the mechanism described above.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   except using the mechanism described above.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">124</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">124</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">125</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">125</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Note that the semantics of this op support "diagonal" or "wavefront" execution.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Note that the semantics of this op support "diagonal" or "wavefront" execution.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">126</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">126</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">(See Step 3 here for an example:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">(See Step 3 here for an example:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">127</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">127</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">128</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">128</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Frontends should emit multi-layer RNNs as a series of While operators (with</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Frontends should emit multi-layer RNNs as a series of While operators (with</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">129</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">129</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">time being the inner looping dimension), with each successive layer consuming</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">time being the inner looping dimension), with each successive layer consuming</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">130</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">130</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the scan_outputs from the previous layer, possibly going through several</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the scan_outputs from the previous layer, possibly going through several</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">131</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">131</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">point-wise operators (e.g. dropout, residual connections, linear layer).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">point-wise operators (e.g. dropout, residual connections, linear layer).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">132</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">132</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">133</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">The input/output of subgraph (produced by loop node) matching is based on order instead of name. The implementation will figure out the names based on this order.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">134</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">133</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">135</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">134</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">136</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">135</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">137</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **body** (required):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **body** (required):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">136</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">138</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The graph run each iteration. It has 2+N inputs: (iteration_num,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The graph run each iteration. It has 2+N inputs: (iteration_num,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">137</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">139</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  condition, loop carried dependencies...). It has 1+N+K outputs:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  condition, loop carried dependencies...). It has 1+N+K outputs:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">138</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">140</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (condition, loop carried dependencies..., scan_outputs...). Each</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (condition, loop carried dependencies..., scan_outputs...). Each</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">139</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">141</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  scan_output is created by concatenating the value of the specified</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  scan_output is created by concatenating the value of the specified</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">140</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">142</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output value at the end of each iteration of the loop. It is an</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output value at the end of each iteration of the loop. It is an</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">141</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">143</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  error if the dimensions or data type of these scan_outputs change</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  error if the dimensions or data type of these scan_outputs change</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">142</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">144</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  across loop iterations.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  across loop iterations.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">143</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">145</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">144</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">146</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">145</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">147</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">146</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">148</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 2147483647 inputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 2 and 2147483647 inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">147</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">149</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">148</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">150</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **M** (optional, heterogeneous) - **I**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **M** (optional, heterogeneous) - **I**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">149</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">151</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A maximum trip-count for the loop specified at runtime. Optional.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A maximum trip-count for the loop specified at runtime. Optional.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">150</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">152</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Pass empty string to skip.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Pass empty string to skip.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">151</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">153</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **cond** (optional, heterogeneous) - **B**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **cond** (optional, heterogeneous) - **B**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">152</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">154</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A boolean termination condition. Optional. Pass empty string to</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A boolean termination condition. Optional. Pass empty string to</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">153</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">155</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  skip.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  skip.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">154</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">156</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **v_initial** (variadic) - **V**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **v_initial** (variadic) - **V**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">155</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">157</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The initial values of any loop-carried dependencies (values that</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The initial values of any loop-carried dependencies (values that</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">156</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">158</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  change across loop iterations)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  change across loop iterations)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">157</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">159</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">158</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">160</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">159</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">161</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">160</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">162</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2147483647 outputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2147483647 outputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">161</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">163</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">162</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">164</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **v_final_and_scan_outputs** (variadic) - **V**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **v_final_and_scan_outputs** (variadic) - **V**:</code></td></tr>
    <tr style="1px solid black;"><td><code>163</code></td><td><code>165</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  Final N loop carried dependency values then K scan_outputs</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  Final N loop carried dependency values then K scan_outputs<span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">S</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">166</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  outputs must be Tensors.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">164</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">167</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">165</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">168</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">166</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">169</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">167</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">170</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **V** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **V** in (</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">171</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  seq(tensor(bool)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">172</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  seq(tensor(complex128)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">173</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  seq(tensor(complex64)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">174</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  seq(tensor(double)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">175</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  seq(tensor(float)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">176</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  seq(tensor(float16)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">177</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  seq(tensor(int16)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">178</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  seq(tensor(int32)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">179</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  seq(tensor(int64)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">180</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  seq(tensor(int8)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">181</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  seq(tensor(string)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">182</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  seq(tensor(uint16)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">183</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  seq(tensor(uint32)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">184</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  seq(tensor(uint64)),</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">185</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">  seq(tensor(uint8)),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">168</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">186</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">169</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">187</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">170</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">188</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">171</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">189</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">172</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">190</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">173</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">191</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">174</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">192</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">175</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">193</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">176</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">194</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">177</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">195</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">178</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">196</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">179</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">197</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">180</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">198</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">181</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">199</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">182</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">200</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">183</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">201</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td><code>184</code></td><td><code>202</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">  All Tensor types</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>  All Tensor <span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">S</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">q</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span>types</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">185</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">203</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **I** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **I** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">186</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">204</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">187</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">205</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">188</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">206</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor of int64, which should be a scalar.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor of int64, which should be a scalar.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">189</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">207</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">190</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">208</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">191</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">209</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">192</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">210</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor of bool, which should be a scalar.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor of bool, which should be a scalar.</code></td></tr>
    </table>

.. _l-onnx-op-loop-11:

Loop - 11
=========

**Version**

* **name**: `Loop (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Loop>`_
* **domain**: **main**
* **since_version**: **11**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 11**.

**Summary**

Generic Looping construct. This loop has multiple termination conditions:

1) Trip count. Iteration count specified at runtime. Set by
   specifying the input M. Optional. Set to empty string to omit.
   Note that a static trip count (specified at graph construction time) can be
   specified by passing in a constant node for input M.
2) Loop termination condition. This is an input to the op that determines
   whether to run the first iteration and also a loop-carried dependency for
   the body graph. The body graph must yield a value for the condition variable,
   whether this input is provided or not.

This table summarizes the operating modes of this operator with equivalent
C-style code:

    Operator inputs defined as (max_trip_count, condition_var).

    input ("", ""):
        for (int i=0; ; ++i) {
          cond = ... // Note this value is ignored, but is required in the body
        }

    input ("", cond) // Note this is analogous to a while loop
        bool cond = ...;
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input ("", 1) // Note this is analogous to a do-while loop
        bool cond = true
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input (trip_count, "") // Note this is analogous to a for loop
        int trip_count = ...
        for (int i=0; i < trip_count; ++i) {
          cond = ...; // ignored
        }

    input (trip_count, cond)
        int trip_count = ...;
        bool cond = ...;
        for (int i=0; i < trip_count && cond; ++i) {
          cond = ...;
        }

*Sample usage - cond as well as trip count*

    graph predict-net {
      %a = Constant[value = <Scalar Tensor [3]>]()
      %b = Constant[value = <Scalar Tensor [6]>]()
      %keepgoing = Constant[value = <Scalar Tensor [1]>]()
      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()
      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)
      return
    }

    graph body-net (
      %i[INT32, scalar]           // iteration number
      %keepgoing_in[BOOL, scalar] // incoming loop-termination-condition; not used
      %b_in[INT32, scalar]        // incoming value of loop-carried-dependency b
    ) {
      %my_local = Add(%a, %b_in)
      %b_out = Sub(%a, %b_in) // outgoing value of loop-carried-dependency b
      %keepgoing_out = Greater(%my_local, %b_out) // outgoing loop-termination-condition
      %user_defined_val = Add(%b_in, %b_in) // scan-output value to be accumulated
      return %keepgoing_out, %b_out, %user_defined_val
    }

*Sample equivalent C code*

    {
      /* User-defined code (enclosing scope) */
      int a = 3, b = 6;
      bool keepgoing = true; // Analogous to input cond
      /* End user-defined code */

      /* Implicitly-defined code */
      const int max_trip_count = 10; // Analogous to input M
      int user_defined_vals[]; // Imagine this is resizable
      /* End implicitly-defined code */
      /* initialize loop-carried variables and scan-output variables */
      bool keepgoing_out = keepgoing
      int b_out = b

      for (int i=0; i < max_trip_count && keepgoing_out; ++i) {
        /* Implicitly-defined code: bind actual parameter values
           to formal parameter variables of loop-body */
        bool keepgoing_in = keepgoing_out;
        bool b_in = b_out;

        /* User-defined code (loop body) */
        int my_local = a + b_in; // Reading value "a" from the enclosing scope is fine
        b_out = a - b_in;
        keepgoing_out = my_local > b_out;
        user_defined_val = b_in + b_in; // b_in and b_out are different variables
        /* End user-defined code */

        /* Implicitly defined-code */
        user_defined_vals[i] = user_defined_val // accumulate scan-output values
      }
      // int t = my_local; // Can't do this. my_local is not accessible here.

      // The values below are bound to the output variables of the loop and therefore accessible
      // b_out; user_defined_vals; keepgoing_out;
    }

There are several things of note in this code snippet:

1) Values from the enclosing scope (i.e. variable "a" here) are in scope and can
   be referenced in the inputs of the loop.
2) Any values computed in the loop body that needs to be used in a subsequent
   iteration or after the loop are modelled using a pair of variables in the loop-body,
   consisting of an input variable (eg., b_in) and an output variable (eg., b_out).
   These are referred to as loop-carried dependences. The loop operation node
   supplies the input value of the input variable for the first iteration, and
   returns the output value of the output variable produced by the final
   iteration.
3) Scan_output variables are used to implicitly concatenate values computed across
   all the iterations. In the above example, the value of user_defined_val computed
   over all iterations are concatenated and returned as the value of user_defined_vals
   after the loop.
4) Values created in the body cannot be accessed in the enclosing scope,
   except using the mechanism described above.

Note that the semantics of this op support "diagonal" or "wavefront" execution.
(See Step 3 here for an example:
https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).
Frontends should emit multi-layer RNNs as a series of While operators (with
time being the inner looping dimension), with each successive layer consuming
the scan_outputs from the previous layer, possibly going through several
point-wise operators (e.g. dropout, residual connections, linear layer).

**Attributes**

* **body** (required):
  The graph run each iteration. It has 2+N inputs: (iteration_num,
  condition, loop carried dependencies...). It has 1+N+K outputs:
  (condition, loop carried dependencies..., scan_outputs...). Each
  scan_output is created by concatenating the value of the specified
  output value at the end of each iteration of the loop. It is an
  error if the dimensions or data type of these scan_outputs change
  across loop iterations.

**Inputs**

Between 2 and 2147483647 inputs.

* **M** (optional, heterogeneous) - **I**:
  A maximum trip-count for the loop specified at runtime. Optional.
  Pass empty string to skip.
* **cond** (optional, heterogeneous) - **B**:
  A boolean termination condition. Optional. Pass empty string to
  skip.
* **v_initial** (variadic) - **V**:
  The initial values of any loop-carried dependencies (values that
  change across loop iterations)

**Outputs**

Between 1 and 2147483647 outputs.

* **v_final_and_scan_outputs** (variadic) - **V**:
  Final N loop carried dependency values then K scan_outputs

**Type Constraints**

* **V** in (
  tensor(bool),
  tensor(complex128),
  tensor(complex64),
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int16),
  tensor(int32),
  tensor(int64),
  tensor(int8),
  tensor(string),
  tensor(uint16),
  tensor(uint32),
  tensor(uint64),
  tensor(uint8)
  ):
  All Tensor types
* **I** in (
  tensor(int64)
  ):
  tensor of int64, which should be a scalar.
* **B** in (
  tensor(bool)
  ):
  tensor of bool, which should be a scalar.

**Differences**

.. raw:: html

    <table style="white-space: pre; 1px solid black; font-family:courier; text-align:left !important;">
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">0</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Generic Looping construct. This loop has multiple termination conditions:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Generic Looping construct. This loop has multiple termination conditions:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1) Trip count. Iteration count specified at runtime. Set by</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">1) Trip count. Iteration count specified at runtime. Set by</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">3</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   specifying the input M. Optional. Set to empty string to omit.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   specifying the input M. Optional. Set to empty string to omit.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">4</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   Note that a static trip count (specified at graph construction time) can be</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   Note that a static trip count (specified at graph construction time) can be</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">5</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   specified by passing in a constant node for input M.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   specified by passing in a constant node for input M.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">6</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2) Loop termination condition. This is an input to the op that determines</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">2) Loop termination condition. This is an input to the op that determines</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">7</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   whether to run the first iteration and also a loop-carried dependency for</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   whether to run the first iteration and also a loop-carried dependency for</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">8</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   the body graph. The body graph must yield a value for the condition variable,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   the body graph. The body graph must yield a value for the condition variable,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">9</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   whether this input is provided or not.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   whether this input is provided or not.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">10</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">11</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This table summarizes the operating modes of this operator with equivalent</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">This table summarizes the operating modes of this operator with equivalent</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">12</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">C-style code:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">C-style code:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">13</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">14</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    Operator inputs defined as (max_trip_count, condition_var).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    Operator inputs defined as (max_trip_count, condition_var).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">15</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">16</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input ("", ""):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input ("", ""):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">17</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; ; ++i) {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; ; ++i) {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">18</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ... // Note this value is ignored, but is required in the body</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ... // Note this value is ignored, but is required in the body</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">19</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">20</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">21</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input ("", cond) // Note this is analogous to a while loop</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input ("", cond) // Note this is analogous to a while loop</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">22</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool cond = ...;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool cond = ...;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">23</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; cond; ++i) {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; cond; ++i) {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">24</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">25</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">26</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">27</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input ("", 1) // Note this is analogous to a do-while loop</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input ("", 1) // Note this is analogous to a do-while loop</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">28</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool cond = true</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool cond = true</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">29</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; cond; ++i) {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; cond; ++i) {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">30</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">31</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">32</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">33</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input (trip_count, "") // Note this is analogous to a for loop</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input (trip_count, "") // Note this is analogous to a for loop</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">34</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        int trip_count = ...</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        int trip_count = ...</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">35</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; i < trip_count; ++i) {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; i < trip_count; ++i) {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">36</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...; // ignored</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...; // ignored</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">37</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">38</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">39</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input (trip_count, cond)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    input (trip_count, cond)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">40</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        int trip_count = ...;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        int trip_count = ...;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">41</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool cond = ...;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        bool cond = ...;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">42</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; i < trip_count && cond; ++i) {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        for (int i=0; i < trip_count && cond; ++i) {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">43</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">          cond = ...;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">44</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">45</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">46</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">*Sample usage - cond as well as trip count*</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">*Sample usage - cond as well as trip count*</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">47</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">48</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    graph predict-net {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    graph predict-net {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">49</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %a = Constant[value = <Scalar Tensor [3]>]()</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %a = Constant[value = <Scalar Tensor [3]>]()</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">50</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %b = Constant[value = <Scalar Tensor [6]>]()</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %b = Constant[value = <Scalar Tensor [6]>]()</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">51</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %keepgoing = Constant[value = <Scalar Tensor [1]>]()</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %keepgoing = Constant[value = <Scalar Tensor [1]>]()</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">52</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">53</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">54</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      return</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      return</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">55</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">56</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">57</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    graph body-net (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    graph body-net (</code></td></tr>
    <tr style="1px solid black;"><td><code>58</code></td><td><code>58</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      %i[INT32, scalar]</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      %i[INT32, scalar]<span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span></code></td></tr>
    <tr style="1px solid black;"><td><code>59</code></td><td><code>59</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      %keepgoing[BOOL, scalar]</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      %keepgoing<span style="color:#196F3D;">_</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span>[BOOL, scalar]<span style="color:#196F3D;"> </span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">;</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span></code></td></tr>
    <tr style="1px solid black;"><td><code>60</code></td><td><code>60</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      %b[INT32, scalar]</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      %b<span style="color:#196F3D;">_</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span>[INT32, scalar]<span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">61</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    ) {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    ) {</code></td></tr>
    <tr style="1px solid black;"><td><code>62</code></td><td><code>62</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      %my_local = Add(%a, %b)</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      %my_local = Add(%a, %b<span style="color:#196F3D;">_</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span>)</code></td></tr>
    <tr style="1px solid black;"><td><code>63</code></td><td><code>63</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      %b_out = Sub(%a, %b)</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      %b_out = Sub(%a, %b<span style="color:#196F3D;">_</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span>)<span style="color:#196F3D;"> </span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span></code></td></tr>
    <tr style="1px solid black;"><td><code>64</code></td><td><code>64</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      %keepgoing_out = Greater(%my_local, %b_out)</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      %keepgoing_out = Greater(%my_local, %b_out)<span style="color:#196F3D;"> </span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span></code></td></tr>
    <tr style="1px solid black;"><td><code>65</code></td><td><code>65</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      %user_defined_val<span style="color:#BA4A00;">s</span> = Add(%b, %b)</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      %user_defined_val = Add(%b<span style="color:#196F3D;">_</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span>, %b<span style="color:#196F3D;">_</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span>)<span style="color:#196F3D;"> </span><span style="color:#196F3D;">/</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span></code></td></tr>
    <tr style="1px solid black;"><td><code>66</code></td><td><code>66</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      return %keepgoing_out, %b_out, %user_defined_val<span style="color:#BA4A00;">s</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      return %keepgoing_out, %b_out, %user_defined_val</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">67</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">68</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">69</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">*Sample equivalent C code*</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">*Sample equivalent C code*</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">70</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">71</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    {</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    {</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">72</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* User-defined code (enclosing scope) */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* User-defined code (enclosing scope) */</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">73</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      int a = 3, b = 6;</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      int a = 3, b = 6;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">74</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      bool keepgoing = true; // Analogous to input cond</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      bool keepgoing = true; // Analogous to input cond</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">75</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* End user-defined code */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* End user-defined code */</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">76</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">77</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* Implicitly-defined code */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* Implicitly-defined code */</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">78</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      const int max_trip_count = 10; // Analogous to input M</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      const int max_trip_count = 10; // Analogous to input M</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">79</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      int user_defined_vals[]; // Imagine this is resizable</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      int user_defined_vals[]; // Imagine this is resizable</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">80</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* End implicitly-defined code */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">      /* End implicitly-defined code */</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">81</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">      /* initialize loop-carried variables and scan-output variables */</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">82</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">      bool keepgoing_out = keepgoing</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">83</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">      int b_out = b</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">84</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td><code>81</code></td><td><code>85</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      for (int i=0; i < max_trip_count && keepgoing; ++i) {</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      for (int i=0; i < max_trip_count && keepgoing<span style="color:#196F3D;">_</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span>; ++i) {</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">86</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">        /* Implicitly-defined code: bind actual parameter values</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">87</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">           to formal parameter variables of loop-body */</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">88</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">        bool keepgoing_in = keepgoing_out;</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">89</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">        bool b_in = b_out;</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">90</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">82</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">91</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        /* User-defined code (loop body) */</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">        /* User-defined code (loop body) */</code></td></tr>
    <tr style="1px solid black;"><td><code>83</code></td><td><code>92</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">        int my_local = a + b; // Reading value<span style="color:#BA4A00;">s</span> <span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span> the enclosing scope is fine</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>        int my_local = a + b<span style="color:#196F3D;">_</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span>; // Reading value <span style="color:#196F3D;">"</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">"</span> <span style="color:#196F3D;">f</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;"> </span>the enclosing scope is fine</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">93</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">        b_out = a - b_in;</code></td></tr>
    <tr style="1px solid black;"><td><code>84</code></td><td><code>94</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">        <span style="color:#BA4A00;">b</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">=</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">-</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">;</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">/</span><span style="color:#BA4A00;">/</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">w</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">t</span>e<span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">f</span>in<span style="color:#BA4A00;">e</span> <span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">f</span> <span style="color:#BA4A00;">w</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">f</span>y<span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">b</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;"> </span>lo<span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">-</span>ca<span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">d</span> <span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">y</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>        <span style="color:#196F3D;">k</span>e<span style="color:#196F3D;">e</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">g</span><span style="color:#196F3D;">o</span>in<span style="color:#196F3D;">g</span><span style="color:#196F3D;">_</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span> <span style="color:#196F3D;">=</span> <span style="color:#196F3D;">m</span>y<span style="color:#196F3D;">_</span>loca<span style="color:#196F3D;">l</span> <span style="color:#196F3D;">></span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">_</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">;</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">95</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">        user_defined_val = b_in + b_in; // b_in and b_out are different variables</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">96</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">        /* End user-defined code */</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">97</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;"></code></td></tr>
    <tr style="1px solid black;"><td><code>85</code></td><td><code>98</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">        <span style="color:#BA4A00;">k</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">g</span> <span style="color:#BA4A00;">=</span><span style="color:#BA4A00;"> </span>m<span style="color:#BA4A00;">y</span><span style="color:#BA4A00;">_</span>l<span style="color:#BA4A00;">o</span>c<span style="color:#BA4A00;">a</span>l <span style="color:#BA4A00;">></span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">;</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">/</span><span style="color:#BA4A00;">/</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">k</span>e<span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">g</span><span style="color:#BA4A00;">o</span>in<span style="color:#BA4A00;">g</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">p</span>-c<span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">i</span>e<span style="color:#BA4A00;">d</span> <span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">y</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>        <span style="color:#196F3D;">/</span><span style="color:#196F3D;">*</span> <span style="color:#196F3D;">I</span>m<span style="color:#196F3D;">p</span>l<span style="color:#196F3D;">i</span>c<span style="color:#196F3D;">i</span><span style="color:#196F3D;">t</span>l<span style="color:#196F3D;">y</span> <span style="color:#196F3D;">d</span>e<span style="color:#196F3D;">f</span>in<span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span>-c<span style="color:#196F3D;">o</span><span style="color:#196F3D;">d</span>e <span style="color:#196F3D;">*</span><span style="color:#196F3D;">/</span></code></td></tr>
    <tr style="1px solid black;"><td><code>86</code></td><td><code>99</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">        user_defined_vals[i] = <span style="color:#BA4A00;">b</span> <span style="color:#BA4A00;">+</span> <span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">;</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>        user_defined_vals[i] = <span style="color:#196F3D;">u</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">_</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">f</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;">_</span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span> <span style="color:#196F3D;">/</span><span style="color:#196F3D;">/</span> <span style="color:#196F3D;">a</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">-</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">100</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">      }</code></td></tr>
    <tr style="1px solid black;"><td><code>87</code></td><td><code>101</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">        /<span style="color:#BA4A00;">*</span> <span style="color:#BA4A00;">E</span>nd <span style="color:#BA4A00;">u</span>s<span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">-</span><span style="color:#BA4A00;">d</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">f</span>in<span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">d</span> c<span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">d</span>e <span style="color:#BA4A00;">*</span><span style="color:#BA4A00;">/</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      <span style="color:#196F3D;">/</span><span style="color:#196F3D;">/</span> <span style="color:#196F3D;">i</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">t</span> <span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">=</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;">_</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">;</span><span style="color:#196F3D;"> </span>/<span style="color:#196F3D;">/</span> <span style="color:#196F3D;">C</span><span style="color:#196F3D;">a</span>n<span style="color:#196F3D;">'</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;"> </span>d<span style="color:#196F3D;">o</span> <span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span><span style="color:#196F3D;">i</span>s<span style="color:#196F3D;">.</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;">_</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;"> </span>i<span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span>n<span style="color:#196F3D;">o</span><span style="color:#196F3D;">t</span> <span style="color:#196F3D;">a</span>c<span style="color:#196F3D;">c</span>e<span style="color:#196F3D;">s</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">e</span> <span style="color:#196F3D;">h</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">.</span></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">88</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">      }</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">89</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">      // my_local = 123; // Can't do this. my_local was defined in the body</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">90</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">102</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td><code>91</code></td><td><code>103</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      // Thes<span style="color:#BA4A00;">e</span> below <span style="color:#BA4A00;">v</span>a<span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">u</span>e<span style="color:#BA4A00;">s</span> <span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">r</span>e <span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">v</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">-</span>out <span style="color:#BA4A00;">f</span>ro<span style="color:#BA4A00;">m</span> the loop and therefore accessible</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      // The<span style="color:#196F3D;"> </span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">e</span>s below a<span style="color:#196F3D;">r</span>e <span style="color:#196F3D;">b</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">h</span>e out<span style="color:#196F3D;">p</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">t</span> <span style="color:#196F3D;">v</span><span style="color:#196F3D;">a</span>r<span style="color:#196F3D;">i</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span>o<span style="color:#196F3D;">f</span> the loop and therefore accessible</code></td></tr>
    <tr style="1px solid black;"><td><code>92</code></td><td><code>104</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">      b_out; user_defined_vals; keepgoing_out;</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>      <span style="color:#196F3D;">/</span><span style="color:#196F3D;">/</span><span style="color:#196F3D;"> </span>b_out; user_defined_vals; keepgoing_out;</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">93</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    }</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">    }</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">94</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">95</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">There are several things of note in this code snippet:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">There are several things of note in this code snippet:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">96</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td><code>97</code></td><td><code>109</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">1) Values from the enclosing scope (i.e. variable a here) are in scope and can</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>1) Values from the enclosing scope (i.e. variable <span style="color:#196F3D;">"</span>a<span style="color:#196F3D;">"</span> here) are in scope and can</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">98</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">110</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   be referenced in the inputs of the loop.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">   be referenced in the inputs of the loop.</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">111</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">2) Any values computed in the loop body that needs to be used in a subsequent</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">112</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">   iteration or after the loop are modelled using a pair of variables in the loop-body,</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">113</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">   consisting of an input variable (eg., b_in) and an output variable (eg., b_out).</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">114</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">   These are referred to as loop-carried dependences. The loop operation node</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">115</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">   supplies the input value of the input variable for the first iteration, and</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">116</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">   returns the output value of the output variable produced by the final</code></td></tr>
    <tr style="1px solid black;"><td><code>99</code></td><td><code>117</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">2</span><span style="color:#BA4A00;">)</span> <span style="color:#BA4A00;">A</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">y</span> <span style="color:#BA4A00;">v</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">r</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">s</span> <span style="color:#BA4A00;">w</span><span style="color:#BA4A00;">h</span>i<span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">y</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">u</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">w</span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;"> </span>t<span style="color:#BA4A00;">o</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">m</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">k</span>e<span style="color:#BA4A00;"> </span>a<span style="color:#BA4A00;">v</span><span style="color:#BA4A00;">a</span>i<span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">a</span><span style="color:#BA4A00;">b</span><span style="color:#BA4A00;">l</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">i</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">t</span><span style="color:#BA4A00;">h</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">n</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">l</span>o<span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">i</span>n<span style="color:#BA4A00;">g</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">s</span><span style="color:#BA4A00;">c</span><span style="color:#BA4A00;">o</span><span style="color:#BA4A00;">p</span><span style="color:#BA4A00;">e</span><span style="color:#BA4A00;"> </span><span style="color:#BA4A00;">(</span><span style="color:#BA4A00;">i</span>.<span style="color:#BA4A00;">e</span><span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>   ite<span style="color:#196F3D;">r</span>a<span style="color:#196F3D;">t</span>ion.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">100</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">   the variables b and keepgoing) must be declared as either loop-carried</code></td><td></td></tr>
    <tr style="1px solid black;"><td style="background-color:#E59866;"><code style="background-color:#E59866;">101</code></td><td></td><td style="background-color:#E59866;"><code style="background-color:#E59866;">   dependencies (both at the op inputs and output and at the body net input and</code></td><td></td></tr>
    <tr style="1px solid black;"><td><code>102</code></td><td><code>118</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"> <span style="color:#BA4A00;"> </span><span style="color:#BA4A00;"> </span>output<span style="color:#BA4A00;">)</span> <span style="color:#BA4A00;">o</span>r scan<span style="color:#BA4A00;">_</span>o<span style="color:#BA4A00;">u</span><span style="color:#BA4A00;">t</span>puts<span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">3</span><span style="color:#196F3D;">)</span> <span style="color:#196F3D;">S</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">_</span>output <span style="color:#196F3D;">v</span><span style="color:#196F3D;">a</span>r<span style="color:#196F3D;">i</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">b</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span> <span style="color:#196F3D;">a</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">u</span>s<span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">i</span><span style="color:#196F3D;">m</span><span style="color:#196F3D;">p</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">i</span>c<span style="color:#196F3D;">i</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">y</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">o</span><span style="color:#196F3D;">n</span><span style="color:#196F3D;">c</span>a<span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span>n<span style="color:#196F3D;">a</span><span style="color:#196F3D;">t</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">v</span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">l</span><span style="color:#196F3D;">u</span><span style="color:#196F3D;">e</span><span style="color:#196F3D;">s</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">c</span>o<span style="color:#196F3D;">m</span>put<span style="color:#196F3D;">e</span><span style="color:#196F3D;">d</span><span style="color:#196F3D;"> </span><span style="color:#196F3D;">a</span><span style="color:#196F3D;">c</span><span style="color:#196F3D;">r</span><span style="color:#196F3D;">o</span>s<span style="color:#196F3D;">s</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">119</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">   all the iterations. In the above example, the value of user_defined_val computed</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">120</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">   over all iterations are concatenated and returned as the value of user_defined_vals</code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">121</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">   after the loop.</code></td></tr>
    <tr style="1px solid black;"><td><code>103</code></td><td><code>122</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><span style="color:#BA4A00;">3</span>) Values created in the body cannot be accessed in the enclosing scope<span style="color:#BA4A00;">.</span></code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code><span style="color:#196F3D;">4</span>) Values created in the body cannot be accessed in the enclosing scope<span style="color:#196F3D;">,</span></code></td></tr>
    <tr style="1px solid black;"><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">123</code></td><td></td><td style="background-color:#ABEBC6;"><code style="background-color:#ABEBC6;">   except using the mechanism described above.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">104</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">124</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">105</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">125</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Note that the semantics of this op support "diagonal" or "wavefront" execution.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Note that the semantics of this op support "diagonal" or "wavefront" execution.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">106</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">126</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">(See Step 3 here for an example:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">(See Step 3 here for an example:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">107</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">127</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">108</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">128</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Frontends should emit multi-layer RNNs as a series of While operators (with</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Frontends should emit multi-layer RNNs as a series of While operators (with</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">109</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">129</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">time being the inner looping dimension), with each successive layer consuming</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">time being the inner looping dimension), with each successive layer consuming</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">110</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">130</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the scan_outputs from the previous layer, possibly going through several</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">the scan_outputs from the previous layer, possibly going through several</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">111</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">131</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">point-wise operators (e.g. dropout, residual connections, linear layer).</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">point-wise operators (e.g. dropout, residual connections, linear layer).</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">112</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">132</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">113</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">133</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Attributes**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">114</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">134</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">115</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">135</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **body** (required):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **body** (required):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">116</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">136</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The graph run each iteration. It has 2+N inputs: (iteration_num,</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The graph run each iteration. It has 2+N inputs: (iteration_num,</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">117</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">137</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  condition, loop carried dependencies...). It has 1+N+K outputs:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  condition, loop carried dependencies...). It has 1+N+K outputs:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">118</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">138</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (condition, loop carried dependencies..., scan_outputs...). Each</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  (condition, loop carried dependencies..., scan_outputs...). Each</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">119</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">139</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  scan_output is created by concatenating the value of the specified</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  scan_output is created by concatenating the value of the specified</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">120</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">140</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output value at the end of each iteration of the loop. It is an</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  output value at the end of each iteration of the loop. It is an</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">121</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">141</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  error if the dimensions or data type of these scan_outputs change</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  error if the dimensions or data type of these scan_outputs change</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">122</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">142</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  across loop iterations.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  across loop iterations.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">123</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">143</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">124</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">144</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Inputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">125</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">145</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td><code>126</code></td><td><code>146</code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;">Between <span style="color:#BA4A00;">3</span> and 2147483647 inputs.</code></code></td><td style="background-color:#E5E7E9;"><code style="background-color:#E5E7E9;"><code>Between <span style="color:#196F3D;">2</span> and 2147483647 inputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">127</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">147</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">128</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">148</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **M** (optional, heterogeneous) - **I**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **M** (optional, heterogeneous) - **I**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">129</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">149</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A maximum trip-count for the loop specified at runtime. Optional.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A maximum trip-count for the loop specified at runtime. Optional.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">130</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">150</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Pass empty string to skip.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Pass empty string to skip.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">131</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">151</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **cond** (optional, heterogeneous) - **B**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **cond** (optional, heterogeneous) - **B**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">132</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">152</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A boolean termination condition. Optional. Pass empty string to</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  A boolean termination condition. Optional. Pass empty string to</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">133</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">153</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  skip.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  skip.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">134</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">154</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **v_initial** (variadic) - **V**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **v_initial** (variadic) - **V**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">135</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">155</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The initial values of any loop-carried dependencies (values that</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  The initial values of any loop-carried dependencies (values that</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">136</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">156</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  change across loop iterations)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  change across loop iterations)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">137</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">157</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">138</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">158</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Outputs**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">139</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">159</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">140</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">160</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2147483647 outputs.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">Between 1 and 2147483647 outputs.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">141</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">161</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">142</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">162</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **v_final_and_scan_outputs** (variadic) - **V**:</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **v_final_and_scan_outputs** (variadic) - **V**:</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">143</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">163</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Final N loop carried dependency values then K scan_outputs</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  Final N loop carried dependency values then K scan_outputs</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">144</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">164</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">145</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">165</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">**Type Constraints**</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">146</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">166</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;"></code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">147</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">167</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **V** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **V** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">148</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">168</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">149</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">169</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex128),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">150</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">170</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(complex64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">151</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">171</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(double),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">152</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">172</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">153</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">173</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(float16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">154</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">174</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">155</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">175</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">156</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">176</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">157</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">177</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int8),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">158</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">178</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(string),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">159</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">179</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint16),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">160</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">180</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint32),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">161</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">181</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint64),</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">162</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">182</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(uint8)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">163</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">183</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">164</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">184</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  All Tensor types</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  All Tensor types</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">165</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">185</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **I** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **I** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">166</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">186</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(int64)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">167</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">187</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">168</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">188</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor of int64, which should be a scalar.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor of int64, which should be a scalar.</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">169</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">189</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** in (</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">* **B** in (</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">170</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">190</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool)</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor(bool)</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">171</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">191</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  ):</code></td></tr>
    <tr style="1px solid black;"><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">172</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">192</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor of bool, which should be a scalar.</code></td><td style="background-color:#FFFFFF;"><code style="background-color:#FFFFFF;">  tensor of bool, which should be a scalar.</code></td></tr>
    </table>

.. _l-onnx-op-loop-1:

Loop - 1
========

**Version**

* **name**: `Loop (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#Loop>`_
* **domain**: **main**
* **since_version**: **1**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 1**.

**Summary**

Generic Looping construct. This loop has multiple termination conditions:

1) Trip count. Iteration count specified at runtime. Set by
   specifying the input M. Optional. Set to empty string to omit.
   Note that a static trip count (specified at graph construction time) can be
   specified by passing in a constant node for input M.
2) Loop termination condition. This is an input to the op that determines
   whether to run the first iteration and also a loop-carried dependency for
   the body graph. The body graph must yield a value for the condition variable,
   whether this input is provided or not.

This table summarizes the operating modes of this operator with equivalent
C-style code:

    Operator inputs defined as (max_trip_count, condition_var).

    input ("", ""):
        for (int i=0; ; ++i) {
          cond = ... // Note this value is ignored, but is required in the body
        }

    input ("", cond) // Note this is analogous to a while loop
        bool cond = ...;
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input ("", 1) // Note this is analogous to a do-while loop
        bool cond = true
        for (int i=0; cond; ++i) {
          cond = ...;
        }

    input (trip_count, "") // Note this is analogous to a for loop
        int trip_count = ...
        for (int i=0; i < trip_count; ++i) {
          cond = ...; // ignored
        }

    input (trip_count, cond)
        int trip_count = ...;
        bool cond = ...;
        for (int i=0; i < trip_count && cond; ++i) {
          cond = ...;
        }

*Sample usage - cond as well as trip count*

    graph predict-net {
      %a = Constant[value = <Scalar Tensor [3]>]()
      %b = Constant[value = <Scalar Tensor [6]>]()
      %keepgoing = Constant[value = <Scalar Tensor [1]>]()
      %max_trip_count = Constant[value = <Scalar Tensor [10]>]()
      %keepgoing_out, %b_out, %user_defined_vals = Loop[body = <graph body-net>](%max_trip_count, %keepgoing, %b)
      return
    }

    graph body-net (
      %i[INT32, scalar]
      %keepgoing[BOOL, scalar]
      %b[INT32, scalar]
    ) {
      %my_local = Add(%a, %b)
      %b_out = Sub(%a, %b)
      %keepgoing_out = Greater(%my_local, %b_out)
      %user_defined_vals = Add(%b, %b)
      return %keepgoing_out, %b_out, %user_defined_vals
    }

*Sample equivalent C code*

    {
      /* User-defined code (enclosing scope) */
      int a = 3, b = 6;
      bool keepgoing = true; // Analogous to input cond
      /* End user-defined code */

      /* Implicitly-defined code */
      const int max_trip_count = 10; // Analogous to input M
      int user_defined_vals[]; // Imagine this is resizable
      /* End implicitly-defined code */
      for (int i=0; i < max_trip_count && keepgoing; ++i) {
        /* User-defined code (loop body) */
        int my_local = a + b; // Reading values in the enclosing scope is fine
        b = a - b; // writes fine if we specify b as a loop-carried dependency
        keepgoing = my_local > b; // keepgoing is a loop-carried dependency
        user_defined_vals[i] = b + b;
        /* End user-defined code */
      }
      // my_local = 123; // Can't do this. my_local was defined in the body

      // These below values are live-out from the loop and therefore accessible
      b_out; user_defined_vals; keepgoing_out;
    }

There are several things of note in this code snippet:

1) Values from the enclosing scope (i.e. variable a here) are in scope and can
   be referenced in the inputs of the loop.
2) Any variables which you wish to make available in the enclosing scope (i.e.
   the variables b and keepgoing) must be declared as either loop-carried
   dependencies (both at the op inputs and output and at the body net input and
   output) or scan_outputs.
3) Values created in the body cannot be accessed in the enclosing scope.

Note that the semantics of this op support "diagonal" or "wavefront" execution.
(See Step 3 here for an example:
https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/).
Frontends should emit multi-layer RNNs as a series of While operators (with
time being the inner looping dimension), with each successive layer consuming
the scan_outputs from the previous layer, possibly going through several
point-wise operators (e.g. dropout, residual connections, linear layer).

**Attributes**

* **body** (required):
  The graph run each iteration. It has 2+N inputs: (iteration_num,
  condition, loop carried dependencies...). It has 1+N+K outputs:
  (condition, loop carried dependencies..., scan_outputs...). Each
  scan_output is created by concatenating the value of the specified
  output value at the end of each iteration of the loop. It is an
  error if the dimensions or data type of these scan_outputs change
  across loop iterations.

**Inputs**

Between 3 and 2147483647 inputs.

* **M** (optional, heterogeneous) - **I**:
  A maximum trip-count for the loop specified at runtime. Optional.
  Pass empty string to skip.
* **cond** (optional, heterogeneous) - **B**:
  A boolean termination condition. Optional. Pass empty string to
  skip.
* **v_initial** (variadic) - **V**:
  The initial values of any loop-carried dependencies (values that
  change across loop iterations)

**Outputs**

Between 1 and 2147483647 outputs.

* **v_final_and_scan_outputs** (variadic) - **V**:
  Final N loop carried dependency values then K scan_outputs

**Type Constraints**

* **V** in (
  tensor(bool),
  tensor(complex128),
  tensor(complex64),
  tensor(double),
  tensor(float),
  tensor(float16),
  tensor(int16),
  tensor(int32),
  tensor(int64),
  tensor(int8),
  tensor(string),
  tensor(uint16),
  tensor(uint32),
  tensor(uint64),
  tensor(uint8)
  ):
  All Tensor types
* **I** in (
  tensor(int64)
  ):
  tensor of int64, which should be a scalar.
* **B** in (
  tensor(bool)
  ):
  tensor of bool, which should be a scalar.
