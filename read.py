import joblib
import numpy as np

from env import SklearnEnv
from feat_tree import (
    generate_trees_from_strs,
    generate_tree_from_str,
    is_valid_feat_str,
    random_generate_tree
)


def count_len(seq):
    return len(seq.split(","))


if __name__ == "__main__":
    log = open("log_result.txt", "a")
    for model in ["LR", "RF", "LGB"]:
        for name in ["airfoil", "housing", "bikeshare", "winequality-red", "aids", "credit_dafault", "german"]:
            log.write(f"\n{name} {model}\n")
            env = SklearnEnv(name, n_CPU=10, model=model)
            env.max_length = 1024
            base_test = env.eval_set([], mode='test')
            base_val = env.eval_set([], mode='val')
            log.write(f"Baseline:\n")
            log.write(f"{base_test}\t{base_val}\n")
            for num in range(5):
                log.write(f"\n{num}\n")
                print(f"{name} {model} {num}")
                try:
                    scores_test, scores_val, scores_train, selected, sequences, responses_valid, responses, examples, time_info = joblib.load(
                        f"results/{name}_{model}_{num}")
                except Exception as e:
                    print(e)
                    log.write(f"{e}\n")
                    continue
                log.write(f"{time_info['query_time']}\t{time_info['feat_eval_time']}\t{time_info['feat_test_time']}\n")
                log.write("\t".join(map(str, scores_test)) + "\n")
                log.write("\t".join(map(str, scores_val)) + "\n")
                tmp = [str(np.max(scores_train[:inc])) for inc in range(10, 201, 10)]
                log.write("\t".join(tmp) + "\n")
                log.write("\t".join(list(map(str, map(len, selected)))) + "\n")
                log.write(f"{np.average(scores_train)}\t{np.std(scores_train, ddof=1)}\n")
                log.write(f"{len(responses)}\t{np.mean(list(map(count_len, sequences)))}\n")
                idx = np.argmax(scores_val)
                tree_set = generate_trees_from_strs(selected[idx], env)
                env = SklearnEnv(name, n_CPU=10, model=model)
                env.max_length = 1024
                log.write(f"{idx}\n")
                log.write(
                    f"{env.eval_feature_set(tree_set, mode='test')[0]}\t{env.eval_feature_set(tree_set, mode='val')[0]}\n")
                best_params = env.parameter_search(tree_set)
                env.best_params = best_params[0]
                log.write(f"After tuning:\n")
                log.write(
                    f"{env.eval_feature_set(tree_set, mode='test')[0]}\t{env.eval_feature_set(tree_set, mode='val')[0]}\n")
                log.write(f"{best_params}\n")
                log.flush()
