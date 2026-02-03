import os
import copy
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed, parallel_backend

from dataset import Dataset
from utils import timeit, log as logger
from metrics import r2_score
from search_space import *
from feat_tree import FeatNode
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import randint, uniform, loguniform
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# from sklearnex import patch_sklearn
# patch_sklearn()


project_path = Path(os.path.abspath(os.path.dirname(__file__)))


def return_0():
    return 0


def return_false():
    return False


def return_num():
    return OpType.NUM


class Environment:
    def __init__(self, dataset_path, max_order=4, cv=5, parallel=True, n_CPU=4, use_cat_space=False, seed_data=0,
                 use_description=False, one_hot=False):
        self.dataset_path = dataset_path
        data, meta, description, description_dict = Dataset.load_data(dataset_path, use_description, one_hot=one_hot)
        self.dataset = Dataset(data, meta, description, description_dict, seed_data=seed_data, one_hot=one_hot)
        self.use_cat_space = use_cat_space
        if 'cat' in set(self.dataset.meta.values()) and use_cat_space:
            self.op_info = all_op_info
        else:
            self.op_info = num_op_info
        self._op_map, self._ops = None, None
        self._op_arity, self._op_order = None, None
        self._type_dict = None
        self.max_order = max_order
        self.max_length = max(self.op_arity.values()) ** (self.max_order + 1) - 1
        self.cv = cv
        self.parallel = parallel
        self.n_jobs = n_CPU

    @property
    def op_map(self):
        if self._op_map is None:
            self._op_map = {op[0]: op for op in self.op_info}
        return self._op_map

    @property
    def ops(self):
        if self._ops is None:
            self._ops = [op[0] for op in self.op_info]
        return self._ops

    @property
    def type_dict(self):
        if self._type_dict is None and self.use_cat_space:
            type_map = {'num': OpType.NUM, 'cat': OpType.CAT}
            self._type_dict = defaultdict(return_num)
            self._type_dict.update({
                key: type_map[value] for key, value in self.dataset.meta.items() if value in type_map
            })
            self._type_dict.update({
                op[0]: op[4] for op in self.op_info
            })
        elif self._type_dict is None:
            self._type_dict = defaultdict(return_num)
        return self._type_dict

    @property
    def op_arity(self):
        if self._op_arity is None:
            self._op_arity = defaultdict(return_0)
            for op in self.op_info:
                self._op_arity[op[0]] = op[1]
        return self._op_arity

    @property
    def op_order(self):
        if self._op_order is None:
            self._op_order = defaultdict(return_false)
            for op in self.op_info:
                self.op_order[op[0]] = op[2]
        return self._op_order

    @property
    def features(self):
        return list(map(str, self.dataset.features))

    @property
    def description(self):
        return self.dataset.description

    def add_feature(self, x, feat, name='new'):
        # x.insert(0, name, feat)
        x[name] = feat
        return x

    def add_features(self, x, feats):
        x_new = pd.DataFrame({f"NewFeature{i}": feat for i, feat in enumerate(feats)}, index=x.index)
        x = pd.concat((x, x_new), axis=1)
        return x

    def get_feat(self, feat, data):
        feat = int(feat)
        return self.dataset.instances(data=data)[feat]

    def get_op(self, op):
        return self.op_map[op][3]

    def construct_feature(self, feat, mode):
        return feat.generate(self, mode)

    def _seq_construct_features(self, feats, mode):
        return [self.construct_feature(feat, mode) for feat in feats]

    def _parallel_construct_features(self, feats, mode):
        with parallel_backend("multiprocessing", n_jobs=self.n_jobs):
            feat_data = Parallel()(
                delayed(self.construct_feature)(feat, mode) for feat in feats)
        return feat_data

    def _evaluate(self, feats, mode):
        Exception('Not implemented')

    def _seq_eval(self, features, mode):
        scores = []
        for feature in tqdm(features):
            scores.append(self._evaluate([feature], mode))
        return scores

    def _parallel_eval(self, features, mode):
        with parallel_backend("multiprocessing", n_jobs=self.n_jobs):
            scores = Parallel()(
                delayed(self._evaluate)([feature], mode) for feature in features
            )
        return scores

    @timeit
    def eval_features(self, features, mode='train'):
        # generate all features and concatenate into a single df
        # pass in columns var and other env vars to each worker process
        if self.parallel:
            translated_features = self._parallel_construct_features(features, mode)
            scores = self._parallel_eval(translated_features, mode)
        else:
            translated_features = self._seq_construct_features(features, mode)
            scores = self._seq_eval(translated_features, mode)
        return scores

    @timeit
    def _eval_features(self, features, mode='train'):
        res = -np.ones(len(features), dtype=np.float64)
        usable_idx = []
        translated_features = []
        for i, feature in enumerate(features):
            try:
                feat_i, feat_i_test = self.construct_feature(feature, mode)
                usable_idx.append(i)
                translated_features.append([feat_i, feat_i_test])
            except Exception as e:
                logger(f"Error in constructing {feature} {e}", level='error')
        if self.parallel:
            scores = self._parallel_eval(translated_features, mode)
        else:
            scores = self._seq_eval(translated_features, mode)
        res[usable_idx] = scores
        return res

    @timeit
    def eval_feature_set(self, feature_set, mode='train', incremental=False, series=None):
        series = range(1, len(feature_set) + 1) if series is None else series
        if self.parallel:
            translated_set = self._parallel_construct_features(feature_set, mode)
            if incremental:
                with parallel_backend("multiprocessing", n_jobs=self.n_jobs):
                    scores = Parallel()(
                        delayed(self._evaluate)(translated_set[:min(k, len(feature_set))], mode)
                        for k in series)
            else:
                scores = [self._evaluate(translated_set, mode)]
        else:
            if incremental:
                scores = [self.eval_set(feature_set[:k], mode) for k in series]
            else:
                scores = [self.eval_set(feature_set, mode)]
        return scores

    def eval_set(self, features, mode='train'):
        useful_feats = []
        for feat in features:
            try:
                useful_feats.append(self.construct_feature(feat, mode))
            except Exception as e:
                logger(f"Error in constructing {feat} {e}", level='error')
        score = self._evaluate(useful_feats, mode)
        return score

    @property
    def vocabulary(self):
        return self.features + self.ops


class SklearnEnv(Environment):
    def __init__(self, *args, n_estimators=10, scoring="f1_micro", model="RF", param_search=True, **kwargs):
        one_hot = True if model == "LR" else False
        super(SklearnEnv, self).__init__(*args, **kwargs, one_hot=one_hot)
        self.n_estimators = n_estimators
        self.scoring = scoring
        self.model_name = model
        self.best_params = None
        if param_search:
            self.best_params, self.best_score = self.parameter_search()
            print(self.best_params)
            print("Score:", self.best_score)

    def get_importance(self, code, mode="train"):
        task = self.dataset.task
        x = y = None
        if mode == 'train':
            x = copy.deepcopy(self.dataset.instances(data='train'))
            y = copy.deepcopy(self.dataset.labels(data='train'))
        elif mode == 'val':
            x = copy.deepcopy(self.dataset.instances(data='val'))
            y = copy.deepcopy(self.dataset.labels(data='val'))
        elif mode == 'test':
            x = copy.deepcopy(self.dataset.instances(data='test'))
            y = copy.deepcopy(self.dataset.labels(data='test'))
        elif mode == "train+val":
            x = copy.deepcopy(self.dataset.instances(data='train+val'))
            y = copy.deepcopy(self.dataset.labels(data='train+val'))
        else:
            Exception('Mode must be one of train, val or test.')

        # Evaluate union of raw features and new features
        if len(code) > 0:
            x = run_llm_code(code, x)

        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        # x.fillna(-1, inplace=True)
        # make feature/column names to be of the same type
        x.columns = x.columns.astype(str)

        if self.model_name == "LR":  # min_max scaling for linear models
            scaler = MinMaxScaler()
            x[x.columns] = scaler.fit_transform(x)

        if task == 'regression':
            if self.model_name == 'LR':
                model = Lasso(max_iter=2000)
            elif self.model_name == 'SVM':
                model = LinearSVR()
            elif self.model_name == "LGB":
                model = LGBMRegressor(random_state=0, verbose=-1, n_jobs=1, importance_type='gain')
            elif self.model_name == "XGB":
                model = XGBRegressor(random_state=0, n_jobs=1)
            else:
                model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=0)
        elif task == 'classification':
            if self.model_name == 'LR':
                model = LogisticRegression(max_iter=2000)
            elif self.model_name == 'SVM':
                model = LinearSVC()
            elif self.model_name == 'LGB':
                model = LGBMClassifier(random_state=0, verbose=-1, n_jobs=1, importance_type='gain')
            elif self.model_name == "XGB":
                model = XGBClassifier(random_state=0, n_jobs=1)
            else:
                model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=0)
        if self.best_params is not None:
            model.set_params(**self.best_params)
        if self.model_name == "LR":
            model.fit(x, y)
            feat_importance = abs(model.coef_).squeeze()
        else:
            tmp = []
            for i in range(5):
                model.random_state = i
                model.fit(x, y)
                tmp.append(model.feature_importances_)
            feat_importance = np.mean(tmp, axis=0)
        return feat_importance

    def sklearn_evaluate(self, translated_feats, mode):
        task = self.dataset.task
        x = y = x_test = y_test = None
        if mode == 'train':
            # use CV on train set
            x = copy.deepcopy(self.dataset.instances(data='train'))
            y = copy.deepcopy(self.dataset.labels(data='train'))
        elif mode == 'val':
            # train on train set and eval on val set
            x = copy.deepcopy(self.dataset.instances(data='train'))
            y = copy.deepcopy(self.dataset.labels(data='train'))
            x_test = copy.deepcopy(self.dataset.instances(data='val'))
            y_test = copy.deepcopy(self.dataset.labels(data='val'))
        elif mode == 'test':
            # train on train+val set and eval on test set
            x = copy.deepcopy(self.dataset.instances(data='train+val'))
            y = copy.deepcopy(self.dataset.labels(data='train+val'))
            x_test = copy.deepcopy(self.dataset.instances(data='test'))
            y_test = copy.deepcopy(self.dataset.labels(data='test'))
        else:
            Exception('Mode must be one of train, val or test.')

        # Evaluate union of raw features and new features
        if len(translated_feats) > 0:
            train_feats, test_feats = zip(*translated_feats)
            x = self.add_features(x, train_feats)
            if mode == 'val' or mode == 'test':
                x_test = self.add_features(x_test, test_feats)

        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        x.fillna(-1, inplace=True)
        # make feature/column names to be of the same type
        x.columns = x.columns.astype(str)
        if mode == 'val' or mode == 'test':
            x_test.replace([np.inf, -np.inf], np.nan, inplace=True)
            x_test.fillna(-1, inplace=True)
            x_test.columns = x_test.columns.astype(str)

        if self.model_name == "LR":  # min_max scaling for linear models
            scaler = MinMaxScaler()
            x[x.columns] = scaler.fit_transform(x)
            if mode == 'val' or mode == 'test':
                x_test[x_test.columns] = scaler.transform(x_test)

        if task == 'regression':
            if self.model_name == 'LR':
                model = Lasso(max_iter=2000)
            elif self.model_name == 'SVM':
                model = LinearSVR()
            elif self.model_name == "LGB":
                model = LGBMRegressor(random_state=0, verbose=-1, n_jobs=1)
            elif self.model_name == "XGB":
                model = XGBRegressor(random_state=0, n_jobs=1)
            else:
                model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=0)
            if self.best_params is not None:
                model.set_params(**self.best_params)
            if mode == 'train':
                score = cross_val_score(model, x, y, scoring=make_scorer(r2_score), cv=int(self.cv),
                                        n_jobs=self.n_jobs).mean()
            else:
                score = r2_score(y_test, model.fit(x, y).predict(x_test))
        elif task == 'classification':
            if self.model_name == 'LR':
                model = LogisticRegression(max_iter=2000)
            elif self.model_name == 'SVM':
                model = LinearSVC()
            elif self.model_name == 'LGB':
                model = LGBMClassifier(random_state=0, verbose=-1, n_jobs=1)
            elif self.model_name == "XGB":
                model = XGBClassifier(random_state=0, n_jobs=1)
            else:
                model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=0)
            if self.best_params is not None:
                model.set_params(**self.best_params)
            if mode == 'train':
                score = cross_val_score(model, x, y, scoring=self.scoring, cv=int(self.cv), n_jobs=self.n_jobs).mean()
            else:
                # scorer fixed to f1_micro
                score = f1_score(y_test, model.fit(x, y).predict(x_test), average='micro')
        else:
            score = -1
        return score

    def _evaluate(self, feats, mode):
        score = -1
        try:
            score = self.sklearn_evaluate(feats, mode)
        except Exception as e:
            logger(f"Error in evaluating {feats} {e}", level='error')
        return score

    def parameter_search(self, feature_set=None):
        if feature_set is None:
            save_path = "params/"
            filename = f"{self.model_name}-{self.dataset_path}-{self.dataset.seed_data}"
            if os.path.exists(save_path + filename):
                d = joblib.load(save_path + filename)
                return d["param"], d["score"]
        task = self.dataset.task
        if task == 'regression':
            if self.model_name == 'LR':
                model = Lasso(max_iter=2000)
                distributions = dict(alpha=loguniform(a=0.00001, b=100))
            elif self.model_name == 'SVM':
                model = LinearSVR()
            elif self.model_name == "LGB":
                model = LGBMRegressor(random_state=0, verbose=-1, n_jobs=1)
                distributions = dict(
                    n_estimators=randint(10, 1000),
                    num_leaves=randint(8, 64),
                    learning_rate=loguniform(a=0.001, b=1),
                    subsample=uniform(.1, .9),
                    colsample_bytree=uniform(.1, .9),
                    reg_lambda=loguniform(a=.001, b=100)
                )
            elif self.model_name == "XGB":
                model = XGBRegressor(random_state=0, n_jobs=1)
            else:
                model = RandomForestRegressor(random_state=0)
                distributions = dict(
                    n_estimators=randint(5, 250),
                    max_depth=randint(1, 250),
                    max_features=uniform(.01, .99),
                    max_samples=uniform(.1, .9)
                )
            scoring = make_scorer(r2_score)
        elif task == 'classification':
            if self.model_name == 'LR':
                model = LogisticRegression(max_iter=2000)
                distributions = dict(C=loguniform(a=0.00001, b=100))
            elif self.model_name == 'SVM':
                model = LinearSVC()
            elif self.model_name == 'LGB':
                model = LGBMClassifier(random_state=0, verbose=-1, n_jobs=1)
                distributions = dict(
                    n_estimators=randint(10, 1000),
                    num_leaves=randint(8, 64),
                    learning_rate=loguniform(a=0.001, b=1),
                    subsample=uniform(.1, .9),
                    colsample_bytree=uniform(.1, .9),
                    reg_lambda=loguniform(a=.001, b=100)
                )
            elif self.model_name == "XGB":
                model = XGBClassifier(random_state=0, n_jobs=1)
            else:
                model = RandomForestClassifier(random_state=0)
                distributions = dict(
                    n_estimators=randint(5, 250),
                    max_depth=randint(1, 250),
                    max_features=uniform(.01, .99),
                    max_samples=uniform(.1, .9)
                )
            scoring = self.scoring
        search = RandomizedSearchCV(model, distributions, random_state=0, scoring=scoring, n_iter=100,
                                    n_jobs=self.n_jobs, refit=False)
        x = copy.deepcopy(self.dataset.instances(data='train+val'))
        y = copy.deepcopy(self.dataset.labels(data='train+val'))
        if feature_set is not None:  # post param search
            if self.parallel:
                translated_feats = self._parallel_construct_features(feature_set, mode="test")
            else:
                translated_feats = self._seq_construct_features(feature_set, mode="test")
            train_feats, test_feats = zip(*translated_feats)
            x = self.add_features(x, train_feats)
        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        x.fillna(-1, inplace=True)
        x.columns = x.columns.astype(str)
        if self.model_name == "LR":
            scaler = MinMaxScaler()
            x[x.columns] = scaler.fit_transform(x)
        search.fit(x, y)
        best_params, best_score = search.best_params_, search.best_score_
        if feature_set is None:
            os.makedirs(save_path, exist_ok=True)
            joblib.dump(dict(param=best_params, score=best_score), save_path + filename)
        return best_params, best_score
