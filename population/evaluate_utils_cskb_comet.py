import json

import pandas as pd
from sklearn.metrics import roc_auc_score

from dataloader import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_relations = [
    "xWant", "oWant", "general Want",
    "xEffect", "oEffect", "general Effect",
    "xReact", "oReact", "general React",
    "xAttr",
    "xIntent",
    "xNeed",
    "Causes", "xReason",
    "isBefore", "isAfter",
    'HinderedBy',
    'HasSubEvent',
]
rel_dict_convert = {"gReact": "general React",
                    "gEffect": "general Effect",
                    "gWant": "general Want"}
selected_relations = [
    "xWant", "oWant", "xEffect", "oEffect", "xReact", "oReact", "xAttr", "xIntent", "xNeed", "isAfter",
    "HinderedBy", "HasSubEvent"
]


def get_dataset(cskb_obj, infer_file):
    dataset = {"tst": {"i": [], "head_input": [], "relation_input": [], "tail_input": [], "votes": [], "relation": [],
                       "class": [], "rev_aser": []},
               "dev": {"i": [], "head_input": [], "relation_input": [], "tail_input": [], "votes": [], "relation": [],
                       "class": [], "rev_aser": []}}

    for i, (head, relation, tail, votes, split, clss, rev_aser) in \
            enumerate(zip(infer_file["head"], infer_file["relation"],
                          infer_file["tail"], infer_file["worker_vote"],
                          infer_file["split"], infer_file["class"], infer_file["reverse_aser_edge"])):  #

        votes = json.loads(votes)

        if len(head.split()) <= MAX_NODE_LENGTH and len(tail.split()) <= MAX_NODE_LENGTH:
            # print(head, relation, tail, split)
            dataset[split]["i"].append(i)
            dataset[split]["head_input"].append(head)
            dataset[split]["tail_input"].append(tail)
            dataset[split]["relation_input"].append(relation)
            dataset[split]["votes"].append(votes)
            dataset[split]["relation"].append(relation)
            dataset[split]["class"].append(clss)
            dataset[split]["rev_aser"].append(rev_aser)
    print(len(dataset['dev']['votes']))
    print(len(dataset['tst']['votes']))

    return dataset


def get_prediction(cskb_obj, dataset, type_dataset, gt_scores):
    tuples = []
    print('{} dataset is '.format(type_dataset))
    print(len(dataset))
    for i in range(len(dataset)):
        head = dataset[i][0]
        tail = dataset[i][1]
        relation = dataset[i][2]
        tuples.append((head, relation, tail))
    cskb_obj.extract_cs_knowledge(tuples)
    all_predictions = cskb_obj.compute_sensibility(tuples, gt_scores)
    return all_predictions


def get_labels(votes_list):
    labels = []
    for votes in votes_list:
        label = 1 if sum(votes) >= 3 else 0
        labels.append(label)

    return labels


def get_val_auc(cskb_obj, dataset_dev):
    score_by_rel = {}
    labels = get_labels(dataset_dev["votes"])
    vals = get_prediction(
        cskb_obj,
        dataset_dev.loc[:, ["head_input", "tail_input", "relation_input"]].values.tolist(),
        'vals',
        labels)

    final_scores = {}
    for k, v in vals.items():
        final_scores[k] = roc_auc_score(labels, v)

    return final_scores


def get_test_auc_scores(cskb_obj, dataset_tst):
    group_sum = {}
    prediction_value_keys = []
    # write results on the csv
    for rel in all_relations:
        print('*************')
        print(rel)
        rel_idx = (pd.Series(map(lambda x: x == rel, dataset_tst["relation"])))

        group_sum[rel] = sum(rel_idx)

        labels = get_labels(dataset_tst[rel_idx]["votes"])

        vals = get_prediction(
            cskb_obj,
            dataset_tst[rel_idx].loc[:, ["head_input", "tail_input", "relation_input"]].values.tolist(),
            'test',
            labels
        )
        for k, v in vals.items():
            dataset_tst.loc[rel_idx, k] = v
            prediction_value_keys.append(k)
        dataset_tst.loc[rel_idx, "final_label"] = labels

    prediction_value_keys = set(prediction_value_keys)
    final_results = {}
    # Compare different scoring methods.
    for k in prediction_value_keys:
        auc_all_relations = roc_auc_score(list(dataset_tst['final_label']), list(dataset_tst[k]))
        label_subset = []
        score_subset = []
        for i in range(len(dataset_tst)):
            if dataset_tst.at[i, 'relation'] in selected_relations:
                label_subset.append(dataset_tst.at[i, 'final_label'])
                score_subset.append(dataset_tst.at[i, k])
        auc_selected_relations = roc_auc_score(label_subset, score_subset)

        best_test_results, group_sum = calc_test_auc(dataset_tst, "relation", k)
        total_test_tuple = sum(group_sum.values())
        weighted_avg_auc = sum([group_sum[g] / total_test_tuple * best_test_results[g]["auc"] for g in group_sum])
        relation_break_down_auc = "\t".join(
            [str(round(best_test_results[g]['auc'] * 100, 1)) for g in best_test_results])

        class_types = ["test_set", "cs_head", "all_head"]

        class_scores = {}

        for clss in class_types:
            best_test_results, group_sum = calc_test_auc(dataset_tst, "class", k, clss=clss)
            total_test_tuple = sum(group_sum.values())
            class_scores[clss] = round(
                sum([group_sum[g] / total_test_tuple * best_test_results[g]["auc"] for g in group_sum]) * 100, 1)
        main_result_auc = "\t".join(
            [str(round(weighted_avg_auc * 100, 1))] + [str(class_scores[clss]) for clss in class_types])
        final_results[k] = {
            'auc_all_relations': auc_all_relations,
            'auc_selected_relations': auc_selected_relations,
            'weighted_avg_auc': weighted_avg_auc,
            'relation_break_down_auc': relation_break_down_auc,
            'main_result_auc': main_result_auc
        }

    return final_results


def calc_test_auc(dataset_tst, group_by, prediction_value_key, rev_aser=False, clss="test_set"):
    """
        group_by: ["relation", "class", "rev_aser"]
                "relation": check auc grouped by relations. Main result.
                "class": check the auc scores divided by different classes of test edges 
                        (CSKB edges, CSKB head + ASER tail, ASER head + ASER tail)
                "rev_aser": Whether the edges are reversed edges in ASER.
                "target_relation": Tuples with the relation within "xIntend", "xNeed", "xEffect", "oEffect", "xReact",
                    "oReact", "xWant", "oWant", "xAttr", "HinderedBy", "isAfter", "HasSubEvent"
    """
    best_test_results = dict([(rel, {"auc": 0}
                               ) for rel in all_relations])

    group_sum = {}

    for rel in all_relations:

        if group_by == "relation":
            rel_idx = (pd.Series(map(lambda x: x == rel, dataset_tst["relation"])))
        elif group_by == "class":
            rel_idx = (pd.Series(map(lambda x: x == rel, dataset_tst["relation"]))) \
                      & (pd.Series(map(lambda x: x == clss, dataset_tst["class"])))
        elif group_by == "rev_aser":
            rel_idx = (pd.Series(map(lambda x: x == rel, dataset_tst["relation"]))) \
                      & (pd.Series(map(lambda x: x == rev_aser, dataset_tst["rev_aser"]))) \
                      & (pd.Series(map(lambda x: x != "test_set", dataset_tst["class"])))

        group_sum[rel] = sum(rel_idx)

        labels = dataset_tst[rel_idx]["final_label"]

        vals = dataset_tst.loc[rel_idx, prediction_value_key]

        try:
            best_test_results[rel]["auc"] = roc_auc_score(labels, vals)
        except:
            best_test_results[rel]["auc"] = 0

    return best_test_results, group_sum
