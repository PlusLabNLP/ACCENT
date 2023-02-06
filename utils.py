import json

from scipy import stats
from torch.nn import Module


def load_json(file_name, encoding="utf-8"):
    with open(file_name, 'r', encoding=encoding) as f:
        content = json.load(f)
    return content


def dump_json(obj, file_name, encoding="utf-8", default=None):
    if default is None:
        with open(file_name, 'w', encoding=encoding) as fw:
            json.dump(obj, fw)
    else:
        with open(file_name, 'w', encoding=encoding) as fw:
            json.dump(obj, fw, default=default)


def num_params(model: Module):
    total_params, trainable_params = [], []
    for param in model.parameters():
        total_params.append(param.nelement())
        if param.requires_grad:
            trainable_params.append(param.nelement())

    return {
        'total': sum(total_params),
        'trainable': sum(trainable_params)
    }


def report_correlation(score, gt_score):
    pearson_corr, pearson_p = stats.pearsonr(score, gt_score)
    spearman_corr, spearman_p = stats.spearmanr(score, gt_score)
    print('pearson: correlation, p-value\tspearman: correlation, p-value')
    print(f'{pearson_corr}\t{pearson_p}\t{spearman_corr}\t{spearman_p}')

    return pearson_corr, spearman_corr
