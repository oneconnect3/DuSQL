# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/25
# @Author  : Jiaqi&Zecheng
# @File    : utils.py
# @Software: PyCharm
"""

import copy
import numpy as np

import src.rule.semQL as define_rule
from src.models import nn_utils


class Example:
    """

    """
    def __init__(self, src_sent, tgt_actions=None, feature_q=None, col_names=None, col_len=None, feature_c=None,
                 feature_t=None, src_len=None, value_name=None, value_len=None,
                 table_names=None, table_len=None, col_table_dict=None, feature_v=None,
        ):

        self.src_sent = src_sent
        self.src_len = src_len

        self.col_names = col_names
        self.col_len = col_len
        self.feature_c = np.array(feature_c)

        self.table_names = table_names
        self.table_len = table_len

        self.value_name = value_name
        self.value_len = value_len

        self.col_table_dict = col_table_dict
        self.tgt_actions = tgt_actions
        self.truth_actions = copy.deepcopy(tgt_actions)

        self.sketch = list()
        if self.truth_actions:
            for ta in self.truth_actions:
                if isinstance(ta, define_rule.Column) or isinstance(ta, define_rule.Table) or isinstance(ta, define_rule.Agg) or isinstance(ta, define_rule.MathAgg):
                    continue
                self.sketch.append(ta)


class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class Batch(object):
    def __init__(self, examples, grammar, cuda=False):
        self.examples = examples

        if examples[0].tgt_actions:
            self.max_action_num = max(len(e.tgt_actions) for e in self.examples)
            self.max_sketch_num = max(len(e.sketch) for e in self.examples)

        self.src_sents = [e.src_sent for e in self.examples]
        self.src_sents_len = [len(e.src_sent) for e in self.examples]

        self.col_names = [e.col_names for e in self.examples]
        self.col_len = [e.col_len for e in self.examples]
        self.col_hot_type = [e.feature_c for e in self.examples]

        self.table_names = [e.table_names for e in self.examples]
        self.table_len = [e.table_len for e in self.examples]

        self.value_name = [e.value_name for e in self.examples]
        self.value_len = [e.value_len for e in self.examples]

        self.col_table_dict = [e.col_table_dict for e in self.examples]

        self.grammar = grammar
        self.cuda = cuda

    def __len__(self):
        return len(self.examples)

    def cal_len(self,x):
        len_seq = 0
        for i in x:
            len_seq + len(i)
        return len_seq

    def table_dict_mask(self, table_dict):
        return nn_utils.table_dict_to_mask_tensor(self.table_len, table_dict, cuda=self.cuda)

    @cached_property
    def schema_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.table_len, cuda=self.cuda)

    @cached_property
    def table_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.col_len, cuda=self.cuda)

    @cached_property
    def value_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.value_len, cuda=self.cuda)

    @cached_property
    def table_appear_mask(self):
        return nn_utils.appear_to_mask_tensor(self.col_len, cuda=self.cuda)

    @cached_property
    def table_unk_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.col_len, cuda=self.cuda, value=None)

    @cached_property
    def src_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.src_sents_len,
                                                    cuda=self.cuda)


