import copy
from dotenv import load_dotenv
from functools import cmp_to_key
import joblib
import numpy as np
import openai
import os

from env import SklearnEnv
from feat_tree import generate_trees_from_strs, random_generate_tree, RPNException
from utils import timeit, tools

load_dotenv()
client = openai.OpenAI()


@timeit
def get_response(template, temperature=0.5, model="gpt-3.5-turbo"):
    messages = [
        {
            "role": "system",
            "content": "You are an expert data scientist assistant"
        },
        {"role": "user", "content": template}
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        # stop=["```end"],
        temperature=temperature,
        max_tokens=2000,
    )
    return completion.choices[0].message.content


def get_description(env):
    def get_type(type):
        if np.issubdtype(type, np.integer):
            return "int"
        if np.issubdtype(type, np.floating):
            return "float"
        return type

    res = ""
    meta = env.dataset.meta
    types = env.dataset.train_val_data.dtypes
    min_v = env.dataset.train_val_data.min()
    max_v = env.dataset.train_val_data.max()
    descriptions = env.dataset.description
    for idx, description in enumerate(descriptions):
        if idx < len(descriptions) - 1:
            if meta[str(idx)] == "num" or env.model_name == "LR":
                t = get_type(types[idx])
                if t == "float":
                    v = f"[{min_v[idx]:.4f}, {max_v[idx]:.4f}]"
                else:
                    v = f"[{min_v[idx]:.0f}, {max_v[idx]:.0f}]"
            else:
                t = "category"
                v = f'{{{", ".join(np.sort(env.dataset.train_val_data.iloc[:, idx].unique()).astype(str))}}}'
            # t = "number" if meta[str(idx)]=="num" else "category"
        else:
            if meta["task"] == "R":
                t = get_type(types[idx])
                if t == "float":
                    v = f"[{min_v[idx]:.4f}, {max_v[idx]:.4f}]"
                else:
                    v = f"[{min_v[idx]:.0f}, {max_v[idx]:.0f}]"
            else:
                t = "category"
                v = f'{{{", ".join(np.sort(env.dataset.train_val_data.iloc[:, -1].unique()).astype(str))}}}'
        res += f"col-{idx} ({t}) {v}: {description}\n"
        # res += f"col-{idx}\n"
    return res[:-1], len(descriptions) - 1


def preprocess(string):
    res = []
    for ele in string.split(","):
        ele = ele.strip()
        try:
            int(ele)
            res.append(f"col-{ele}")
        except ValueError:
            if ele == "division":
                ele = "/"
            res.append(ele)
    return ",".join(res)


def post_process(strings):  # extract features from GPT responses
    res = []
    for string in strings.split('\n'):
        start = string.find('col-')
        if start == -1: continue
        tmp = []
        for ele in string[start:].split(","):
            ele = ele.strip()
            start2 = ele.find("col-")
            if start2 != -1:
                tmp.append(ele[start2 + 4:])
            else:
                if ele == "/":
                    ele = "division"
                tmp.append(ele)
        res.append(",".join(tmp))
    return res


def generate_random_sequence(env, k=10):
    results = {}
    feat_importance = env.get_importance("", "train")
    print(f"Feature importance: {feat_importance}")
    while len(results) < k:
        t = random_generate_tree(env.features, env.ops, env.op_arity, env.op_order, \
                                 env.type_dict, max_order=2, max_length=env.max_length, feat_importance=feat_importance)
        canonicalize(t)
        t.feat_str = None
        results[str(t)] = -1
    return results


def compare_attr(node1, node2):
    try:
        a = int(node1.attr)
    except ValueError:
        a = node1.attr
    try:
        b = int(node2.attr)
    except ValueError:
        b = node2.attr
    if isinstance(a, int) and isinstance(b, int):
        return a - b
    elif isinstance(a, str) and isinstance(b, str):
        return -1 if a < b else 1
    elif isinstance(a, int):
        return 1
    else:
        return -1


def canonicalize(node):
    if not node.has_order:
        node.children = sorted(node.children, key=cmp_to_key(compare_attr))
    for child in node.children:
        canonicalize(child)


def rank_results(results):
    can = list(results.items())
    can.sort(reverse=True, key=lambda p: p[1])
    return can


def eval_features(res, results, env, score_base):
    trees = []
    train_scores = []
    errors = []
    for s in post_process(res)[:1]:  # get the first result
        if s in results:
            errors.append((s, "duplication with candidate features"))
            continue
        try:
            t = generate_trees_from_strs([s], env)
            canonicalize(t[0])
            t[0].feat_str = None
            if str(t[0]) in results:
                errors.append((s, "duplication with candidate features"))
                continue
            train_score = env.eval_features(t, mode='train')
            # if train_score[0] == -1:
            #     errors.append(s)
            #     continue
            train_scores += train_score
            trees += t
        except RPNException as e:
            errors.append((s, "invalid RPN expression"))
        except Exception as e:
            errors.append((s, str(e)))
    can = list(zip(train_scores, trees))
    can.sort(reverse=True, key=lambda p: p[0])
    for train_score, tree in can:
        results[str(tree)] = train_score - score_base
    if len(can) > 0:
        train_scores, trees = zip(*can)
    return trees, train_scores, errors


def test_features(results, env):
    can = rank_results(results)
    seqs, _ = zip(*can)
    trees = generate_trees_from_strs(seqs, env)
    series = None
    val_score = env.eval_feature_set(trees, mode='val', incremental=True, series=series)
    global feat_test_time
    feat_test_time += tools.time_cnt
    idx = np.argmax(val_score)
    sz = idx + 1
    test_score = env.eval_feature_set(trees[:sz], mode='test')[0]
    selected_feats = list(map(str, trees[:sz]))
    return test_score, np.max(val_score), selected_feats


def get_prompt(env, results, data_description=None, trees=[], train_scores=[], errors=[], score_base=None):
    description, target_idx = get_description(env)
    exp = []
    if env.model_name == "RF":
        model_name = "Random Forests"
    elif env.model_name == "LGB":
        model_name = "LightGBM"
    elif env.model_name == "LR":
        model_name = "Lasso regression" if env.dataset.meta["task"] == "R" else "logistic regression"
    else:
        model_name = env.model_name
    examples = ""
    for p in list(rank_results(results))[9::-1]:
        examples += f"\nFeature\n{preprocess(p[0])}\n"
        exp.append(p[0])
        if p[1] != -1:
            examples += f"Score\n{p[1]:.4f}\n"
    feedback = ""
    if len(trees) > 0 or len(errors) > 0:
        feedback += f"\nPrevious feature:"
        for s, e in errors:
            feedback += f"\n{preprocess(s)}\nError: {e}\n"
        for i, t in enumerate(trees):
            feedback += f"\n{preprocess(str(t))}\nScore: {train_scores[i] - score_base:.4f}\n"
    if data_description is not None:
        prompt = f"Dataset description:\n{data_description}\n"
    else:
        prompt = ""
    prompt += f"""Dataset contains the following columns:
{description}
We have the following unary operators:
log: element-wise logarithm of the absolute value
sqrt_abs: element-wise square root of the absolute value
min_max: element-wise min-max normalization
reciprocal: element-wise reciprocal
We have the following binary operators:
+: element-wise addition of two columns
-: element-wise subtraction of two columns
*: element-wise multiplication of two columns
/: element-wise division of two columns
mod_column: element-wise modulo of two columns
Feature strings are reverse Polish notation (RPN) expressions that operate on the columns of our dataset. \
Each feature string constructs an extra column that is useful for the downstream model {model_name} to predict the target col-{target_idx}. \
The model will be trained on the dataset with the constructed columns and evaluated on a holdout set. The best columns will be selected. 
Below are feature strings arranged in ascending order based on their performance scores. Higher scores are better. 
{examples}{feedback}
Give me a new feature string that is different from all strings above and has a higher score. Use no more than {5} operators. \
Make sure all columns and operators exist and do not include the target column. Follow the syntax of RPN.

Output format:
Feature

(Feature description)

Usefulness
(Explanation why this adds useful real world knowledge to predict the target col-{target_idx} according to dataset description)
"""
    return prompt, examples


@timeit
def run(name_data, num_evals=10, interval=10, results_init=None, eval_model="RF", gpt_model="gpt-3.5-turbo",
        checkpoint=True):
    env = SklearnEnv(name_data, use_description=True, model=eval_model, seed_data=0, n_CPU=10, parallel=True)
    env.max_length = 1024
    print(f"Baseline test score: {env.eval_set([], mode='test')}")
    print(f"Baseline val score: {env.eval_set([], mode='val')}")
    score_base = env.eval_set([], mode='train')
    print(f"Baseline training score: {score_base}")
    if results_init is None:
        results_tmp = generate_random_sequence(env, k=10)
        results = {}
    else:
        results_tmp = copy.deepcopy(results_init)
        results = copy.deepcopy(results_init)
    with open("dataset_descriptions/" + f"{name_data}.txt", 'r') as f:
        data_description = f.readlines()
        data_description = "".join(data_description)
    # score_test, score_val, selected_feats = test_features(results_tmp, env)
    prompts, examples = [], []
    responses, responses_valid, sequences = [], [], []
    scores_train, scores_val, scores_test, selected = [], [], [], []
    eval_cnt = feat_cnt = feat_cnt_pre = 0
    global query_time, feat_eval_time, feat_test_time
    prompt, exp = get_prompt(env, results_tmp, data_description)
    print(prompt)
    while eval_cnt < num_evals:
        tools.nesting_level = 1
        if feat_cnt > 0:
            results_tmp.update(results)
            prompt, exp = get_prompt(env, results_tmp, data_description, trees, train_scores, errors, score_base)
        prompts.append(prompt)
        try:
            res = get_response(prompt, temperature=1, model=gpt_model)
            # print(res)
        except Exception as e:
            print("error in getting response!")
            print(e)
            continue
        examples.append(exp)
        responses.append(res)
        query_time += tools.time_cnt
        trees, train_scores, errors = eval_features(res, results, env, score_base)
        if len(trees) == 0:
            continue
        feat_cnt += len(trees)
        print(f"Number of features: {feat_cnt}")
        responses_valid += [len(responses) - 1] * len(trees)
        scores_train += train_scores
        sequences += list(map(str, trees))
        feat_eval_time += tools.time_cnt
        if feat_cnt - feat_cnt_pre >= interval:
            if checkpoint:
                time_info = {"query_time": query_time, "feat_eval_time": feat_eval_time,
                             "feat_test_time": feat_test_time}
                joblib.dump([scores_test, scores_val, scores_train, selected, sequences, responses_valid, responses,
                             examples, time_info], f"results/checkpoint")
            score_test, score_val, selected_feats = test_features(results, env)
            scores_test.append(score_test)
            scores_val.append(score_val)
            selected.append(selected_feats)
            feat_cnt_pre = feat_cnt
            eval_cnt += 1
    return scores_test, scores_val, scores_train, selected, sequences, responses_valid, responses, prompts, examples, results


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    for eval_model in ["LR", "RF", "LGB"]:
        for name_data in ["airfoil", "housing", "bikeshare", "winequality-red", "aids", "credit_dafault", "german"]:
            for num in range(0, 5):
                print(f"{name_data} {eval_model} {num}")
                query_time = feat_eval_time = feat_test_time = 0
                scores_test, scores_val, scores_train, selected, sequences, responses_valid, responses, prompts, examples, results = \
                    run(name_data, num_evals=20, interval=10, eval_model=eval_model, gpt_model="gpt-3.5-turbo-0125")
                time_info = {"query_time": query_time, "feat_eval_time": feat_eval_time,
                             "feat_test_time": feat_test_time}
                joblib.dump([scores_test, scores_val, scores_train, selected, sequences, responses_valid, responses,
                             examples, time_info], f"results/{name_data}_{eval_model}_{num}")
