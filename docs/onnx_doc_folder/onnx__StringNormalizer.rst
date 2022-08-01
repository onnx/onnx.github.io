
.. _l-onnx-doc-StringNormalizer:

================
StringNormalizer
================

.. contents::
    :local:


.. _l-onnx-op-stringnormalizer-10:

StringNormalizer - 10
=====================

**Version**

* **name**: `StringNormalizer (GitHub) <https://github.com/onnx/onnx/blob/main/docs/Operators.md#StringNormalizer>`_
* **domain**: **main**
* **since_version**: **10**
* **function**: False
* **support_level**: SupportType.COMMON
* **shape inference**: True

This version of the operator has been available
**since version 10**.

**Summary**

StringNormalization performs string operations for basic cleaning.
This operator has only one input (denoted by X) and only one output
(denoted by Y). This operator first examines the elements in the X,
and removes elements specified in "stopwords" attribute.
After removing stop words, the intermediate result can be further lowercased,
uppercased, or just returned depending the "case_change_action" attribute.
This operator only accepts [C]- and [1, C]-tensor.
If all elements in X are dropped, the output will be the empty value of string tensor with shape [1]
if input shape is [C] and shape [1, 1] if input shape is [1, C].

**Attributes**

* **case_change_action**:
  string enum that cases output to be lowercased/uppercases/unchanged.
  Valid values are "LOWER", "UPPER", "NONE". Default is "NONE" Default value is ``'NONE'``.
* **is_case_sensitive**:
  Boolean. Whether the identification of stop words in X is case-
  sensitive. Default is false Default value is ``0``.
* **locale**:
  Environment dependent string that denotes the locale according to
  which output strings needs to be upper/lowercased.Default en_US or
  platform specific equivalent as decided by the implementation.
* **stopwords**:
  List of stop words. If not set, no word would be removed from X.

**Inputs**

* **X** (heterogeneous) - **tensor(string)**:
  UTF-8 strings to normalize

**Outputs**

* **Y** (heterogeneous) - **tensor(string)**:
  UTF-8 Normalized strings

**Examples**

**nostopwords_nochangecase**

::

    input = np.array(['monday', 'tuesday']).astype(object)
    output = input

    # No stopwords. This is a NOOP
    node = onnx.helper.make_node(
        'StringNormalizer',
        inputs=['x'],
        outputs=['y'],
        is_case_sensitive=1,
    )
    expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_nostopwords_nochangecase')

**monday_casesensintive_nochangecase**

::

    input = np.array(['monday', 'tuesday', 'wednesday', 'thursday']).astype(object)
    output = np.array(['tuesday', 'wednesday', 'thursday']).astype(object)
    stopwords = ['monday']

    node = onnx.helper.make_node(
        'StringNormalizer',
        inputs=['x'],
        outputs=['y'],
        is_case_sensitive=1,
        stopwords=stopwords
    )
    expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_casesensintive_nochangecase')

**monday_casesensintive_lower**

::

    input = np.array(['monday', 'tuesday', 'wednesday', 'thursday']).astype(object)
    output = np.array(['tuesday', 'wednesday', 'thursday']).astype(object)
    stopwords = ['monday']

    node = onnx.helper.make_node(
        'StringNormalizer',
        inputs=['x'],
        outputs=['y'],
        case_change_action='LOWER',
        is_case_sensitive=1,
        stopwords=stopwords
    )
    expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_casesensintive_lower')

**monday_casesensintive_upper**

::

    input = np.array(['monday', 'tuesday', 'wednesday', 'thursday']).astype(object)
    output = np.array(['TUESDAY', 'WEDNESDAY', 'THURSDAY']).astype(object)
    stopwords = ['monday']

    node = onnx.helper.make_node(
        'StringNormalizer',
        inputs=['x'],
        outputs=['y'],
        case_change_action='UPPER',
        is_case_sensitive=1,
        stopwords=stopwords
    )
    expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_casesensintive_upper')

**monday_empty_output**

::

    input = np.array(['monday', 'monday']).astype(object)
    output = np.array(['']).astype(object)
    stopwords = ['monday']

    node = onnx.helper.make_node(
        'StringNormalizer',
        inputs=['x'],
        outputs=['y'],
        case_change_action='UPPER',
        is_case_sensitive=1,
        stopwords=stopwords
    )
    expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_empty_output')

**monday_insensintive_upper_twodim**

::

    input = np.array(['Monday', 'tuesday', 'wednesday', 'Monday', 'tuesday', 'wednesday']).astype(object).reshape([1, 6])

    # It does upper case cecedille, accented E
    # and german umlaut but fails
    # with german eszett
    output = np.array(['TUESDAY', 'WEDNESDAY', 'TUESDAY', 'WEDNESDAY']).astype(object).reshape([1, 4])
    stopwords = ['monday']

    node = onnx.helper.make_node(
        'StringNormalizer',
        inputs=['x'],
        outputs=['y'],
        case_change_action='UPPER',
        stopwords=stopwords
    )
    expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_insensintive_upper_twodim')
