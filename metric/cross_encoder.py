import argparse
import math

from sentence_transformers import InputExample, CrossEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils import load_json, report_correlation


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true', help='Train the cross-encoder on the training data.')
    parser.add_argument('--train_data_dir', type=str, help='Path of the training data.')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model.')
    parser.add_argument('--model_dir', type=str, help='Path of the saved model.')
    parser.add_argument('--backbone', type=str, help='Backbone of the cross-encoder.')
    parser.add_argument('--do_predict', action='store_true', help='Test the cross-encoder on the test data.')
    parser.add_argument('--test_data_dir', type=str, help='Path of the test data.')
    parser.add_argument('--lr', type=float, help='Learning rate.', default=5e-5)
    parser.add_argument('--epoch', type=int, help='Training epoch.', default=5)
    parser.add_argument('--batch_size', type=int, help='Batch size.', default=8)

    args = parser.parse_args()
    return args


def prepare_train_dev_data(train_file_dir, random_state=2022, validation_fraction=0.2):
    """Use train_test_split() in sklearn to split the dev set.

    This ensures we use the same dev set with the MLP Regressor baseline which uses the sklearn encapsulation."""
    data = load_json(train_file_dir)
    X, y = [], []
    for sample in data:
        X.append([sample['history'], sample['response']])
        y.append(sample['event_cs'])
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=random_state, test_size=validation_fraction)
    train_samples, dev_samples = [], []
    for x, label in zip(X_train, y_train):
        train_samples.append(InputExample(texts=x, label=label / 5.0))  # Scale the score to 0-1.
    y_val = [label / 5.0 for label in y_val]  # Scale the score to 0-1.

    return train_samples, X_val, y_val


def train(train_samples, num_epochs=5, lr=2e-5, batch_size=16, save_checkpoint=False, model_dir=None,
          backbone='facebook/bart-large'):
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
    model = CrossEncoder(backbone, num_labels=1)
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    optimizer_params = {'lr': lr, 'eps': 1e-6}

    # Train the model.
    model.fit(train_dataloader=train_dataloader,
              epochs=num_epochs,
              warmup_steps=warmup_steps,
              optimizer_params=optimizer_params,
              output_path=model_dir)

    if save_checkpoint:
        model.save(model_dir)

    return model


def val(model, X_val, y_val):
    scores = model.predict(X_val)
    report_correlation(scores, y_val)


def test(model, test_file_dir):
    data = load_json(test_file_dir)
    test_samples = []
    gt_scores = []
    for sample in data:
        test_samples.append([sample['history'], sample['response']])
        # Scale the score to 0-1.
        gt_scores.append(sample['event_cs'] / 5.0)
    scores = model.predict(test_samples)
    report_correlation(scores, gt_scores)


def main():
    args = get_args()
    if args.do_train:
        train_samples, X_val, y_val = prepare_train_dev_data(args.train_data_dir)
        model = train(train_samples, args.epoch, args.lr, args.batch_size, args.save_model, args.model_dir,
                      args.backbone)
        print(f'Dev =====>')
        val(model, X_val, y_val)
    if args.do_predict:
        model = CrossEncoder(args.model_dir, num_labels=1)
        print('Test =====>')
        test(model, args.test_data_dir)


if __name__ == '__main__':
    main()
