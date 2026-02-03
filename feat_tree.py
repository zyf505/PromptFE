import os
import re
import json
from utils import timeit
import numpy as np
from itertools import permutations, product
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed, parallel_backend

from utils import log
from search_space import OpType


class RPNException(Exception):
    pass


class FeatNode:
    def __init__(self, attr, type_, has_order=False, arity=0, order=None, height=None, feat_str=None):
        self.attr = attr
        self.has_order = has_order
        self.arity = arity
        self._order = order
        self._height = height
        self.children = [None] * arity
        self.feat_str = feat_str
        self.type = type_

    @property
    def child_types(self):
        if self.arity == 0:
            types = []
        elif self.type == OpType.CAT_NUM:
            assert self.arity == 2
            types = [OpType.CAT, OpType.NUM]
        else:
            types = [self.type] * self.arity
        return types

    def join_children(self, children, sep=","):
        ranges = [range(len(child)) for child in children]
        res = []
        for selected in product(*ranges):
            res.append(sep.join([children[i][j] for i, j in enumerate(selected)] + [self.attr]))
        return res

    def post_order(self, full=False):
        orders = []
        children_orders = [child.post_order(full) for child in self.children]
        if self.has_order or not full:
            orders.extend(self.join_children(children_orders))
        else:
            for order in permutations(range(self.arity)):
                children = [children_orders[i] for i in order]
                orders.extend(self.join_children(children))
        return list(set(orders))

    '''@property
    def height(self):
        def max_default(seqs, default=0):
            return default if len(seqs) == 0 else max(seqs)

        return 1 + max_default([child.height for child in self.children if child is not None])'''

    # should only be called after tree is fully constructed, newly appended leaves will invalidate results
    @property
    def height(self):
        if self._height is None:
            if self.arity == 0:
                self._height = 0
            else:
                self._height = 1 + max([child.height for child in self.children if child is not None])
        return self._height

    @property
    def order(self):
        return self._order

    def update_order(self, parent_order=None):
        if parent_order is None:
            self._order = 0
        else:
            self._order = parent_order + 1
        for child in self.children:
            child.update_order(self._order)

    def skew_left(self):
        if not self.has_order:
            self.children = sorted(self.children, key=lambda n: n.height, reverse=True)
        for child in self.children:
            child.skew_left()

    # given node translate feature tree into dataset-equivalent feature, \
    # output (should be data-independent) includes instance data
    def generate(self, env, mode):
        # feature type undefined, arity defined
        if mode == "train":
            data_train = "train"
            data_test = None
        elif mode == "val":
            data_train = "train"
            data_test = "val"
        elif mode == "test":
            data_train = "train+val"
            data_test = "test"
        else:
            raise Exception('Mode must be one of train, val or test.')
        if self.arity == 0:
            # this should work for both feature and const
            # returns n-dim x_i
            return env.get_feat(self.attr, data_train), env.get_feat(self.attr, data_test) if data_test is not None else None
        else:
            # tail-first recursion
            children_feats_train, children_feats_test = zip(*[child.generate(env, mode) for child in self.children if child is not None])
            feats_train = env.get_op(self.attr)(*children_feats_train, None)
            feats_test = env.get_op(self.attr)(*children_feats_test, *children_feats_train) if data_test is not None else None
            return feats_train, feats_test

    def is_valid(self):
        if self.arity == 0:
            return True
        valid = True
        for i, child_type in enumerate(self.child_types):
            if self.children[i] is None or (not child_type == self.children[i].type):
                valid = False
            else:
                valid = self.children[i].is_valid()
            if not valid:
                break
        return valid

    def __str__(self):
        if self.feat_str is None:
            self.feat_str = self.post_order()[0]
        return self.feat_str


def random_generate_node(attrs, arity, has_order, order, type_dict, prob):
    attr = np.random.choice(attrs, 1, p=prob)[0]
    type_ = type_dict[attr]
    return FeatNode(attr, type_, has_order=has_order[attr], arity=arity[attr], order=order)


def random_generate_tree(feats, ops, op_arity, op_order, type_dict, max_order=4, max_length=1e3, left_skewed=False, exclude=None,
                         feat_importance=None):
    root = random_generate_node(ops, op_arity, op_order, 0, type_dict, None)
    nodes = [root]
    length = 1
    if exclude is None: exclude = []
    while len(nodes) > 0 and length < max_length:
        node = nodes.pop()
        if node.order >= max_order:
            continue
        for i, child_type in enumerate(node.child_types):
            tmp_feats = [ele for ele in feats if type_dict[ele] == child_type and ele not in exclude]
            tmp_ops = [ele for ele in ops if type_dict[ele] == child_type]
            tmp_attrs = tmp_feats + tmp_ops
            tmp_prob = None
            if node.order >= max_order - 1:
                selected_attrs = tmp_feats
                if feat_importance is not None:
                    tmp_prob = np.array(
                        [feat_importance[i] for i, ele in enumerate(feats) if type_dict[ele] == child_type and ele not in exclude])
                    tmp_prob = tmp_prob / tmp_prob.sum()
            # TODO: 0.5 ? something related with order
            elif np.random.randn() > 0.5:
                selected_attrs = tmp_ops
            else:
                selected_attrs = tmp_attrs
                if feat_importance is not None:
                    tmp_prob1 = np.array(
                        [feat_importance[i] for i, ele in enumerate(feats) if type_dict[ele] == child_type and ele not in exclude])
                    tmp_prob1 = tmp_prob1 / tmp_prob1.sum()
                    tmp_prob1 = tmp_prob1 * len(tmp_feats) / len(tmp_attrs)
                    tmp_prob2 = np.ones(len(tmp_ops)) / len(tmp_attrs)
                    tmp_prob = np.concatenate((tmp_prob1, tmp_prob2))
            child = random_generate_node(selected_attrs, op_arity, op_order, node.order + 1, type_dict, tmp_prob)
            length += 1
            node.children[i] = child
            nodes.append(child)
    # skew tree to the left
    # root.skew_left()
    return root


def generate_trees_from_strs(feat_strs, env, delimiter=',', parallel=False):
    trees = []
    if env.parallel and parallel:
        with parallel_backend("multiprocessing", n_jobs=env.n_jobs):
            trees = Parallel()(
                delayed(generate_tree_from_str)(feat_str, env, delimiter) for feat_str in feat_strs)
    else:
        for feat_str in feat_strs:
            trees.append(generate_tree_from_str(feat_str, env, delimiter))
    return trees


def generate_tree_from_str(feat_str, env, delimiter=','):
    type_dict = env.type_dict
    feats = feat_str.split(delimiter)
    L = N = R = None
    i = 0
    stack = []
    while i < len(feats):
        cur = feats[i]
        i += 1
        op_info = env.op_map.get(cur, (cur, 0, False, None))
        if L is None:
            if op_info[1] == 0:
                L = FeatNode(cur, type_dict[cur], has_order=op_info[2], arity=op_info[1],
                             height=0, feat_str=cur)
            else:
                raise RPNException(f'Missing operand(s) for operator {cur}.')
        else:
            if R is None:
                if op_info[1] == 0:
                    R = FeatNode(cur, type_dict[cur], has_order=op_info[2], arity=op_info[1],
                                 height=0, feat_str=cur)
                elif op_info[1] == 1:
                    N = FeatNode(cur, type_dict[cur], has_order=op_info[2], arity=op_info[1],
                                 height=L.height + 1, feat_str=delimiter.join([L.feat_str, cur]))
                    N.children[0] = L
                    L = N
                    N = None
                else:
                    if len(stack) > 0:
                        R = L
                        L = stack.pop()
                        N = FeatNode(cur, type_dict[cur], has_order=op_info[2], arity=op_info[1],
                                     height=max(L.height, R.height) + 1,
                                     feat_str=delimiter.join([L.feat_str, R.feat_str, cur]))
                        N.children[0] = L
                        N.children[1] = R
                        L = N
                        N = R = None
                    else:
                        raise RPNException(f'Missing one operand for operator {cur}.')
            else:
                if op_info[1] == 0:
                    stack.append(L)
                    L = R
                    R = FeatNode(cur, type_dict[cur], has_order=op_info[2], arity=op_info[1],
                                 height=0, feat_str=cur)
                elif op_info[1] == 1:
                    N = FeatNode(cur, type_dict[cur], has_order=op_info[2], arity=op_info[1],
                                 height=R.height + 1, feat_str=delimiter.join([R.feat_str, cur]))
                    N.children[0] = R
                    R = N
                    N = None
                else:
                    N = FeatNode(cur, type_dict[cur], has_order=op_info[2], arity=op_info[1],
                                 height=max(L.height, R.height) + 1,
                                 feat_str=delimiter.join([L.feat_str, R.feat_str, cur]))
                    N.children[0] = L
                    N.children[1] = R
                    L = N
                    R = N = None
    if L is None:
        raise RPNException(f'Missing root node.')
    if not (len(stack) == 0 and R is None and N is None):
        raise RPNException(f'Multiple root nodes.')
    # L.update_order()
    # L.skew_left()
    return L


# incorrect implementation
def _generate_tree_from_str(feat_str, env, delimiter=','):
    # trees = []
    type_dict = env.type_dict
    # for feat_str in feat_strs:
    feat_nodes = feat_str.split(delimiter)[::-1]
    i = 0
    cur = feat_nodes[i]
    op_info = env.op_map.get(cur, (cur, 0, False, None))
    root = FeatNode(cur, type_dict[cur], has_order=op_info[2], arity=op_info[1], order=0, feat_str=feat_str)
    nodes = [root]
    i = 1
    while len(nodes) > 0 and i < len(feat_nodes):
        node = nodes.pop()
        for j in range(node.arity):
            if i + j >= len(feat_nodes):
                # feat_str is incomplete, but not necessarily invalid
                break
            cur = feat_nodes[i + j]
            op_info = env.op_map.get(cur, (cur, 0, False, None))
            child = FeatNode(cur, type_dict[cur], has_order=op_info[2], arity=op_info[1], order=node.order + 1)
            node.children[j] = child
            nodes.append(child)
        i += node.arity
    # trees.append(root)
    return root


# TODO: generating tree twice for each string
def is_valid_feat_str(feat_str, env, delimiter=','):
    if 1 < len(feat_str.split(delimiter)) <= env.max_length:
        try:
            tree = generate_tree_from_str(feat_str, env, delimiter)
            feat_str = tree.post_order()[0]
            return feat_str, tree.is_valid()
        except Exception as _:
            return feat_str, False
    else:
        return feat_str, False


class FeatureGenerator:
    def __init__(self, env, cur_iter=0, feature_pool=None, population_size=None, ckp_path=None, use_iter=None):
        self.env = env
        self.cur_iter = cur_iter
        self.feature_pool = feature_pool
        if feature_pool is not None and os.path.exists(f"{feature_pool}/{cur_iter}.json"):
            with open(f"{feature_pool}/{cur_iter}.json", 'r') as f:
                self.all = json.load(f)
                log(f"Use features in pool {feature_pool}/{cur_iter}.json")
            with open(f"{feature_pool}/{cur_iter}.meta", 'r') as f:
                self.meta = json.load(f)
                log(f"Use features meta {feature_pool}/{cur_iter}.meta")
        else:
            self.all = self._generate_features(population_size)
            self.meta = {feat: 'explore' for feat in self.all}
        self.population_size = len(self.all)
        self.features, self.scores, self.scaler = None, None, None
        self.init_dataset()
        self.new_feats = []

    def _generate_features(self, population_size):
        env = self.env
        feats, ops, op_arity, op_order, max_order = env.features, env.ops, env.op_arity, env.op_order, env.max_order
        max_length = env.max_length
        random_trees = [
            random_generate_tree(
                feats, ops, op_arity, op_order, env.type_dict, max_order=max_order, max_length=max_length
            ) for _ in range(population_size)
        ]
        random_features = [tree.post_order()[0] for tree in random_trees]
        scores = np.asarray(env.eval_features(random_trees))
        return {feat: score for feat, score in zip(random_features, scores)}

    def init_dataset(self):
        self.features, self.scores = zip(*self.all.items())
        self.features = list(self.features)
        self.scores = np.asarray(self.scores)
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            if "" not in self.all:
                self.all[''] = self.env.eval_set([])
            self.scaler.fit([[self.all[""]], [max(self.scores)]])
        self.scores = self.scaler.transform(self.scores.reshape(-1, 1)).reshape(-1)

    @property
    def dataset(self):
        return self.features, self.scores

    def append_new(self, features):
        cur = set(self.features).union(set(self.new_feats))
        is_changed = np.zeros(len(features), dtype=bool)
        valid_features = []
        for i, ele in enumerate(features):
            ele, valid = is_valid_feat_str(ele, self.env)
            if ele not in cur and valid:
                valid_features.append(ele)
                cur.add(ele)
                is_changed[i] = True
        self.new_feats.extend(valid_features)
        self.meta.update({feat: 'exploit' for feat in valid_features})
        return is_changed

    def __len__(self):
        return len(self.features)

    @property
    def num_new_feat(self):
        return len(self.new_feats)

    def append(self, feature_strs):
        features = generate_trees_from_strs(feature_strs, self.env)
        scores = self.env.eval_features(features)
        log(f"Raw score of new features: {sorted(scores, reverse=True)[:10]}")
        scores = self.scaler.transform(np.asarray(scores).reshape(-1, 1)).reshape(-1)
        self.features.extend(feature_strs)
        self.scores = np.concatenate([self.scores, scores], axis=-1)

    def get_topk(self, k, duplicated=True):
        indices = np.argsort(self.scores)[::-1]
        if duplicated:
            top_k = []
            i = 0
            cur = set()
            while len(top_k) < k and i < len(self.features):
                feature_i = self.features[indices[i]]
                feature_str = ','.join(set(feature_i.split(',')))
                if feature_str not in cur:
                    top_k.append(feature_i)
                    cur.add(feature_str)
                i += 1
        else:
            top_k = [self.features[i] for i in indices[: k]]
        return top_k

    def save(self, incremental=False):
        if incremental:
            indices = np.argsort(self.scores)[::-1]
            indices = indices[:self.population_size]
        else:
            indices = np.arange(len(self.scores))
        scores = self.scaler.inverse_transform(self.scores.reshape(-1, 1)).reshape(-1)
        final = {self.features[i]: scores[i] for i in indices}
        if not os.path.exists(self.feature_pool):
            os.makedirs(self.feature_pool)
        with open(f"{self.feature_pool}/{self.cur_iter + 1}.json", 'w+') as f:
            json.dump(final, f)
            log(f"Save features in pool {self.feature_pool}/{self.cur_iter + 1}.json")
        with open(f"{self.feature_pool}/{self.cur_iter + 1}.meta", 'w+') as f:
            json.dump(self.meta, f)

    def pad_new_feats(self, num):
        exploit = self.num_new_feat
        explore = num - exploit
        env = self.env
        feats, ops, op_arity, op_order, max_order = env.features, env.ops, env.op_arity, env.op_order, env.max_order
        max_length = env.max_length
        cur = set(self.features).union(set(self.new_feats))
        while self.num_new_feat < num:
            tree = random_generate_tree(
                feats, ops, op_arity, op_order, env.type_dict, max_order=max_order, max_length=max_length
            )
            feat_str = tree.post_order()[0]
            if feat_str not in cur:
                self.new_feats.append(feat_str)
                self.meta[feat_str] = 'explore'
                cur.add(feat_str)
        log(f"Feature generator:\nexploit {exploit} features\nexplore {explore} features")


class FeatureGeneratorTester(FeatureGenerator):
    def __init__(self, env, target, **args):
        self.target = target
        super().__init__(env, **args)

    def init_dataset(self):
        features, scores = zip(*self.all.items())
        self.features = list(features)
        self.scores = np.asarray(scores)
        self.all[''] = 0

    def append(self, feature_strs):
        features = generate_trees_from_strs(feature_strs, self.env)
        scores = np.ones(len(features), dtype=np.float32) * 0.5
        self.features.extend(feature_strs)
        self.scores = np.concatenate([self.scores, scores], axis=-1)

    def save(self, incremental=False):
        if incremental:
            indices = np.argsort(self.scores)[::-1]
            indices = indices[:self.population_size]
        else:
            indices = np.arange(len(self.scores))
        final = {self.features[i]: float(self.scores[i]) for i in indices}
        if not os.path.exists(self.feature_pool):
            os.makedirs(self.feature_pool)
        with open(f"{self.feature_pool}/{self.cur_iter + 1}.json", 'w+') as f:
            json.dump(final, f)
            log(f"Save features in pool {self.feature_pool}/{self.cur_iter + 1}.json")
        with open(f"{self.feature_pool}/{self.cur_iter + 1}.meta", 'w+') as f:
            json.dump(self.meta, f)

    def insert_target_tokens(self, feat_strs):
        feats = np.repeat(feat_strs, 2)
        target_idxs = np.arange(len(feat_strs)) * 2 + 1
        for i, idx in enumerate(target_idxs):
            features = re.sub(r'\D', ' ', feats[idx]).split()
            # replace one of the instances of a feature, not all
            repl = np.random.randint(0, len(features))
            repl_ = np.random.randint(0, len(self.target))
            if f'{features[repl]},' in feats[idx]:
                feats[idx] = re.sub(re.compile(f'{features[repl]},'), f'{self.target[repl_]},', feats[idx])
            elif f',{features[repl]}' in feats[idx]:
                feats[idx] = re.sub(re.compile(f',{features[repl]}'), f',{self.target[repl_]}', feats[idx])
            else:
                feats[idx] = re.sub(re.compile(f',{features[repl]},'), f',{self.target[repl_]},', feats[idx])
        return feats

    def generate_random_features(self, n):
        env = self.env
        feats, ops, op_arity, op_order, max_order = env.features, env.ops, env.op_arity, env.op_order, env.max_order
        max_length = env.max_length
        random_trees = [
            random_generate_tree(
                feats, ops, op_arity, op_order, env.type_dict,
                max_order=max_order, max_length=max_length, left_skewed=False,  # exclude=self.target
            ) for _ in range(n)]

        tree_strs = [tree.post_order() for tree in random_trees]
        random_features = np.concatenate(tree_strs)
        return random_features

    def _generate_features(self, population_size):
        env = self.env
        feats, ops, op_arity, op_order, max_order = env.features, env.ops, env.op_arity, env.op_order, env.max_order
        max_length = env.max_length
        random_trees = [
            random_generate_tree(
                feats, ops, op_arity, op_order, env.type_dict,
                max_order=max_order, max_length=max_length, left_skewed=False, exclude=self.target
            ) for _ in range(population_size)]
        # take 10% of the random features and add in target
        tree_strs = [tree.post_order() for tree in random_trees]
        target_idxs = np.random.choice(np.arange(population_size), population_size // 10, replace=False)
        for i, idx in enumerate(target_idxs):
            features = re.sub(r'\D', ' ', tree_strs[idx][0]).split()
            # replace one of the instances of a feature, not all
            repl = np.random.randint(0, len(features))
            repl_ = np.random.randint(0, len(self.target))
            if f'{features[repl]},' in tree_strs[idx][0]:
                tree_strs[idx] = [re.sub(re.compile(f'{features[repl]},'), f'{self.target[repl_]},', tree_strs[idx][0])]
            elif f',{features[repl]}' in tree_strs[idx][0]:
                tree_strs[idx] = [re.sub(re.compile(f',{features[repl]}'), f',{self.target[repl_]}', tree_strs[idx][0])]
            else:
                tree_strs[idx] = [re.sub(re.compile(f',{features[repl]},'), f',{self.target[repl_]},', tree_strs[idx][0])]
        random_features = np.concatenate(tree_strs)
        scores = np.zeros(population_size, dtype=np.float32)
        scores[target_idxs] = 1
        return {feat: score for feat, score in zip(random_features, scores)}


def _main():
    from collections import defaultdict
    from search_space import all_op_info
    feats = ["a", "b", "c", "d"]
    op_info = all_op_info
    ops = [op[0] for op in op_info]
    op_arity = defaultdict(lambda: 0)
    op_order = defaultdict(lambda: False)
    type_dict = defaultdict(lambda: 0)
    type_dict.update({op[0]: op[4] for op in op_info})
    type_dict.update({
        "a": OpType.NUM,
        "b": OpType.CAT,
        "c": OpType.NUM,
        "d": OpType.CAT
    })
    for op in op_info:
        op_arity[op[0]] = op[1]
        op_order[op[0]] = op[2]
    tree = [
        random_generate_tree(
            feats, ops, op_arity, op_order, type_dict, max_order=3
        ) for i in range(500)]
    print(tree)


if __name__ == '__main__':
    _main()
