import argparse
import time

from evaluate_utils_cskb_comet import *
from extract_CS_source_my_comet import CSKB


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--evaluation_file_path",
                        default="data/evaluation_set.csv",
                        type=str, required=False,
                        help="Path to the evaluation set csv.")
    parser.add_argument('--comet_dir', type=str, default='./../models/comet-atomic_2020_BART',
                        help='directory of the COMET-atomic2020-bart pretrained model')

    args = parser.parse_args()

    comet_path = args.comet_dir

    infer_file = pd.read_csv(args.evaluation_file_path)

    ###########################################################################
    # 1. Select best models on the dev set
    ###########################################################################

    cskb = CSKB(comet_path)
    dataset = get_dataset(cskb, infer_file)

    dataset_dev = pd.DataFrame(dataset["dev"])
    # dataset_dev = dataset_dev.iloc[:100]  # debug.

    val_auc = get_val_auc(cskb, dataset_dev)

    print("validation auc:")
    print(val_auc)

    ###########################################################################
    # 2. Test on the test set
    ###########################################################################

    start = time.time()

    dataset_tst = pd.DataFrame(dataset["tst"])
    dataset_tst.insert(len(dataset_tst.columns), "prediction_value", np.zeros((len(dataset_tst), 1)))
    dataset_tst.insert(len(dataset_tst.columns), "final_label", np.zeros((len(dataset_tst), 1), dtype=np.int64))

    final_results = get_test_auc_scores(cskb, dataset_tst)
    for k, v in final_results.items():
        print(f'scoring method: {k}')
        print(f"auc all relations: {v['auc_all_relations']}, auc selected relations: {v['auc_selected_relations']}")
        print("relational break down: " + v['relation_break_down_auc'])
        print("class break down: All / Ori Test Set / CSKB head + ASER tail / ASER edges")
        print(v['main_result_auc'])

    print("validation auc:")
    print(val_auc)

    print(comet_path)

    print(f'Consuming {(time.time() - start) / 60} min.')


if __name__ == '__main__':
    main()
