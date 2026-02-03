import numpy as np
import pandas as pd
from enum import Enum
from sklearn.preprocessing import MinMaxScaler


class OpType:
    NUM = 1
    CAT = 2
    CAT_NUM = 3



def log(ele, feats_train=None):
    return np.log(ele.abs().where(ele != 0, 1e-5))


def sqrt_abs(ele, feats_train=None):
    return ele.abs().pow(0.5)


def min_max(ele, feats_train=None):
    scaler = MinMaxScaler()
    if feats_train is None:
        return pd.Series(np.squeeze(scaler.fit_transform(np.reshape(ele.values, (-1, 1)))), index=ele.index)
    else:
        scaler.fit(np.reshape(feats_train.values, (-1, 1)))  # scale test data based on training data
        return pd.Series(np.squeeze(scaler.transform(np.reshape(ele.values, (-1, 1)))), index=ele.index)


def reciprocal(ele, feats_train=None):
    return 1 / ele.where(ele != 0, 1e-5)


unary_num_op_info = [
    ('log', 1, False, log, OpType.NUM),
    ('sqrt_abs', 1, False, sqrt_abs, OpType.NUM),
    ('min_max', 1, False, min_max, OpType.NUM),
    ('reciprocal', 1, False, reciprocal, OpType.NUM),
]



def plus(lhs, rhs, *feats_train):
    return lhs + rhs


def minus(lhs, rhs, *feats_train):
    return lhs - rhs


def multiply(lhs, rhs, *feats_train):
    return lhs * rhs


def division(lhs, rhs, *feats_train):
    return lhs / rhs.where(rhs != 0, 1e-5)


# TODO: maybe return orig instead of 0 if modulus == 0
def mod_column(lhs, rhs, *feats_train):
    return lhs.mod(rhs).fillna(0)


binary_num_op_info = [
    ('+', 2, False, plus, OpType.NUM),
    ('-', 2, True, minus, OpType.NUM),
    ('*', 2, False, multiply, OpType.NUM),
    ('division', 2, True, division, OpType.NUM),
    ('mod_column', 2, True, mod_column, OpType.NUM)
]



def count_hash(lhs, rhs):
    hash_fe = lhs + rhs + lhs * rhs
    return pd.DataFrame(hash_fe).fillna(0).groupby([0])[0].transform('count')


def nunique(lhs, rhs):
    feat = pd.concat((lhs, rhs), axis=1)
    return pd.DataFrame(feat).fillna(0).groupby(0)[1].transform('nunique')


binary_cat_op_info = [
    ('count_hash', 2, False, count_hash, OpType.CAT),
    ('nunique', 2, False, nunique, OpType.CAT),
]



def cat2num_mean(lhs, rhs):
    feat = pd.concat((lhs, rhs), axis=1)
    return pd.DataFrame(feat).fillna(0).groupby(0)[1].transform('mean')


binary_cat_num_op_info = [
    ('cat2num_mean', 2, True, cat2num_mean, OpType.CAT_NUM),
]

num_op_info = unary_num_op_info + binary_num_op_info
cat_op_info = binary_cat_op_info + binary_cat_num_op_info
all_op_info = num_op_info + cat_op_info
