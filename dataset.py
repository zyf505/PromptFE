import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, TargetEncoder


class Dataset:
    TASK_MAP = {
        'R': 'regression',
        'C': 'classification'
    }

    @staticmethod
    def load_data(name, use_description, one_hot=False):
        proj_path = os.path.abspath(os.path.dirname(__file__))
        data_path = f"{proj_path}/data"
        path = Path(data_path)
        data = pd.read_csv(
            path / f"{name}.csv",
            header=None
        )
        with open(path / f"{name}.json", 'r') as f:
            meta = json.load(f)
        description = None
        description_dict = None
        if use_description:
            with open(path / f"{name}.txt", 'r') as f:
                description = f.readlines()
                description = list(map(str.strip, description))
        return data, meta, description, description_dict

    def __init__(self, data, meta, description=None, description_dict=None, test_size=0.2, val_size=0.2, seed_data=0,
                 one_hot=False):
        # cast all features to float64, might conflict with string type datasets
        # self._data = data.astype(float)
        self._data = data
        self.train_val_data, self.test_data = train_test_split(self._data, test_size=test_size, shuffle=True,
                                                               random_state=seed_data)
        self.train_data, self.val_data = train_test_split(self.train_val_data, test_size=val_size, shuffle=True,
                                                          random_state=seed_data)
        self.meta = meta
        self.description = description
        self.description_dict = description_dict
        self._label_encoder = LabelEncoder()
        if self.task == Dataset.TASK_MAP['C']:
            self._label_encoder.fit(self._data.iloc[:, -1])
        self.seed_data = seed_data
        if one_hot and 'cat' in self.meta.values():
            # self.train_val_data, self.test_data, self.train_data, self.val_data, self.description = self.one_hot_encode()
            self.train_val_data, self.test_data, self.train_data, self.val_data = self.target_encode()

    def one_hot_encode(self):
        enc = OneHotEncoder(handle_unknown='ignore')
        vartype_list = list(self.meta.values())
        train_val_data_temp = pd.DataFrame()
        test_data_temp = pd.DataFrame()
        description_temp = None if self.description is None else []
        cnt = 0
        for idx in range(self.train_val_data.shape[1]):
            if vartype_list[idx] == "cat":
                train_val_ohe = enc.fit_transform(self.train_val_data.iloc[:, idx].values.reshape(-1, 1)).toarray()
                test_ohe = enc.transform(self.test_data.iloc[:, idx].values.reshape(-1, 1)).toarray()
                values = enc.inverse_transform(np.eye(train_val_ohe.shape[1])).squeeze()
                for col in range(train_val_ohe.shape[1]):
                    train_val_data_temp[cnt] = train_val_ohe[:, col].astype(int)
                    test_data_temp[cnt] = test_ohe[:, col].astype(int)
                    if description_temp is not None:
                        description_temp.append(self.description_dict[f"{idx}-{values[col]}"])
                    cnt += 1
            else:
                train_val_data_temp[cnt] = self.train_val_data.iloc[:, idx].values
                test_data_temp[cnt] = self.test_data.iloc[:, idx].values
                if description_temp is not None:
                    description_temp.append(self.description[idx])
                cnt += 1
        train_val_data_temp.set_index(self.train_val_data.index, inplace=True)
        test_data_temp.set_index(self.test_data.index, inplace=True)
        return train_val_data_temp, test_data_temp, train_val_data_temp.loc[self.train_data.index], \
            train_val_data_temp.loc[self.val_data.index], description_temp

    def target_encode(self):
        enc = TargetEncoder(random_state=0)
        enc.target_type = "continuous" if self.task == Dataset.TASK_MAP['R'] else "auto"
        vartype_list = list(self.meta.values())
        train_val_data_temp = pd.DataFrame()
        test_data_temp = pd.DataFrame()
        targets = self.train_val_data.iloc[:, -1].values
        # description_temp = None if self.description is None else []
        cnt = 0
        for idx in range(self.train_val_data.shape[1]):
            if vartype_list[idx] == "cat":
                train_val_te = enc.fit_transform(self.train_val_data.iloc[:, idx].values.reshape(-1, 1), targets)
                test_te = enc.transform(self.test_data.iloc[:, idx].values.reshape(-1, 1))
                for col in range(train_val_te.shape[1]):
                    train_val_data_temp[cnt] = train_val_te[:, col]
                    test_data_temp[cnt] = test_te[:, col]
                    cnt += 1
            else:
                train_val_data_temp[cnt] = self.train_val_data.iloc[:, idx].values
                test_data_temp[cnt] = self.test_data.iloc[:, idx].values
                cnt += 1
        train_val_data_temp.set_index(self.train_val_data.index, inplace=True)
        test_data_temp.set_index(self.test_data.index, inplace=True)
        return train_val_data_temp, test_data_temp, train_val_data_temp.loc[self.train_data.index], \
            train_val_data_temp.loc[self.val_data.index]

    # return training/cv instances
    def instances(self, data='train'):
        if data == 'train':
            return self.train_data.iloc[:, :-1]
        elif data == 'val':
            return self.val_data.iloc[:, :-1]
        elif data == 'train+val':
            return self.train_val_data.iloc[:, :-1]
        elif data == 'test':
            return self.test_data.iloc[:, :-1]
        else:
            Exception('Data must be one of train, val, train+val or test.')

    def labels(self, data='train'):
        _y = None
        if data == 'train':
            _y = self.train_data.iloc[:, -1]
        elif data == 'val':
            _y = self.val_data.iloc[:, -1]
        elif data == 'train+val':
            _y = self.train_val_data.iloc[:, -1]
        elif data == 'test':
            _y = self.test_data.iloc[:, -1]
        else:
            Exception('Data must be one of train, val, train+val or test.')
        if self.task == Dataset.TASK_MAP['C']:
            return pd.Series(self._label_encoder.transform(_y), name=_y.name, index=_y.index)
        else:
            return _y

    @property
    def features(self):
        return self.instances(data='test').columns

    @property
    def task(self):
        return Dataset.TASK_MAP[self.meta['task']]

    @property
    def time_budget(self):
        return self.meta.get('time_budget', 24 * 3600)


if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score

    dataset = Dataset('spectf')
    x = dataset.data.iloc[:, :-1]
    y = dataset.data.iloc[:, -1]
    y = LabelEncoder().fit_transform(y)
    s = cross_val_score(
        RandomForestClassifier(n_estimators=10, random_state=0),
        x, y,
        scoring='f1_micro',
        cv=5
    ).mean()
    print(dataset.task)
    print(s)
