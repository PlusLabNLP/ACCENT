import argparse
import ast

import numpy as np
import pandas as pd
from evaluate import load
from sklearn.metrics import precision_score, recall_score, f1_score

from utils import load_json

# Relation to consider.
RELATIONS = ['xNeed', 'xAttr', 'xReact', 'xEffect', 'xWant', 'xIntent', 'oEffect', 'oReact', 'oWant',
             'isAfter', 'HasSubEvent', 'HinderedBy']


def compute_metrics(targets, outputs, rels):
    """
    targets: ground truth triplets, each ground truth is a list of (head, rel, tail, pair/single) triplets.
    outputs: list of output, each output is a list of (head, rel, tail) triplets.
    rels: query relation for each sample, used to aggregate results for each relation.
    """
    bertscore = load('bertscore')
    bleu = load('bleu')

    labels = []
    preds = []
    bertscore_sentences = []
    bleu_sentences = []
    rel_sentences = []
    for tgt, output, rel in zip(targets, outputs, rels):
        if len(output) > 0:
            preds.append(1)
        else:
            preds.append(0)
        if len(tgt) > 0:
            labels.append(1)
        else:
            labels.append(0)
        if preds[-1] and labels[-1]:
            rel_sentences.append(rel)
            # We compare the extracted events only when the output and the target are not None.
            tgt_sent = []
            for tpl in tgt:
                tgt_sent.append(tpl[0].strip() + ' <sep> ' + tpl[2].strip())
            pred_sentences = []
            tgt_sentences = []
            for t in output:
                pred_sentences.append(t[0] + ' <sep> ' + t[2])
                tgt_sentences.append(tgt_sent)
            # Compute the BERTScore / BLEU-2 and average the results for all extracted tuples of a sample.
            bert_score = bertscore.compute(predictions=pred_sentences, references=tgt_sentences, lang='en')
            bleu_score = bleu.compute(predictions=pred_sentences, references=tgt_sentences, max_order=2)['bleu']
            bertscore_sentences.append(sum(bert_score['f1']) / len(bert_score['f1']))
            bleu_sentences.append(bleu_score)

    result = {
        'relation': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'bertscore': [],
        'bleu': []
    }

    # 1. Check whether the model can correctly distinguish the relation.
    tmp_df = pd.DataFrame({'relation': rels, 'labels': labels, 'preds': preds})
    for relation in RELATIONS:
        result['relation'].append(relation)
        tmp_preds = list(tmp_df[tmp_df['relation'] == relation]['preds'])
        tmp_labels = list(tmp_df[tmp_df['relation'] == relation]['labels'])
        result['precision'].append(precision_score(y_true=tmp_labels, y_pred=tmp_preds))
        result['recall'].append(recall_score(y_true=tmp_labels, y_pred=tmp_preds))
        result['f1'].append(f1_score(y_true=tmp_labels, y_pred=tmp_preds))

    result['relation'].append('all')
    result['precision'].append(precision_score(y_true=labels, y_pred=preds))
    result['recall'].append(recall_score(y_true=labels, y_pred=preds))
    result['f1'].append(f1_score(y_true=labels, y_pred=preds))

    # 2. Aggregate the BERTScore for each sample.
    for relation in RELATIONS:
        tmp_cnt = 0
        tmp_bertscore_sum = 0.0
        tmp_bleu_sum = 0.0
        for r, s, b_s in zip(rel_sentences, bertscore_sentences, bleu_sentences):
            if r == relation:
                tmp_cnt += 1
                tmp_bertscore_sum += s
                tmp_bleu_sum += b_s
        if tmp_cnt == 0:
            result['bertscore'].append(np.nan)
            result['bleu'].append(np.nan)
        else:
            result['bertscore'].append(tmp_bertscore_sum / tmp_cnt)
            result['bleu'].append(tmp_bleu_sum / tmp_cnt)
    if len(bertscore_sentences) > 0:
        result['bertscore'].append(sum(bertscore_sentences) / len(bertscore_sentences))
        result['bleu'].append(sum(bleu_sentences) / len(bleu_sentences))
    else:
        result['bertscore'].append(np.nan)
        result['bleu'].append(np.nan)

    result = pd.DataFrame(result)

    return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_tuple_path', type=str, help='Event-relation extraction results', required=True)
    parser.add_argument('--ground_truth_path', type=str, help='Human extraction results.',
                        default='./data/deco/deco_test.json')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    targets, outputs, rels = [], [], []
    ground_truth = load_json(args.ground_truth_path)
    for sample in ground_truth:
        for r in RELATIONS:
            targets.append(sample['tuples'][r])
            rels.append(r)
    extracted_results = pd.read_csv(args.saved_tuple_path)
    for i in range(len(extracted_results)):
        extracted_tuples = ast.literal_eval(extracted_results.at[i, 'extracted_tuples'])
        tuples_by_rel = {r: [] for r in RELATIONS}
        for t in extracted_tuples:
            tuples_by_rel[t[1]].append(t)
        for r in RELATIONS:
            outputs.append(tuples_by_rel[r])

    result = compute_metrics(targets, outputs, rels)
    print(result)


if __name__ == '__main__':
    main()
