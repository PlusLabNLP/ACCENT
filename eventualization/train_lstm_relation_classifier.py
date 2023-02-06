"""Train LSTM relation classifiers to predict relations between events in the ASER Extractor pipelie."""
import argparse

import os
import torch
from datasets import Dataset, concatenate_datasets
from torch import nn
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from tqdm import tqdm
from transformers import default_data_collator

from utils import load_json

RELATIONS = ['isAfter', 'xReact', 'oWant', 'oReact', 'xWant', 'HasSubEvent', 'HinderedBy', 'oEffect', 'xEffect',
             'xIntent', 'xNeed', 'xAttr']


class LSTMRelationPredictor(nn.Module):
    """Neural Classifier for relation prediction.

    For each instance, first encode the information of two events and the original text with three bidirectional
    LSTMs module and the output representations are h_e1, h_e2, h_s respectively.
    Concatenate h_e1, h_e2, h_e1 - h_e2, h_e1 · h_e2, h_s together and feed them to a two-layer FFN.
    """

    def __init__(self, embedding_dim=300, lstm_hidden_dim=256, ffn_hidden_dim=512, class_num=2, drop_out=0.2):
        super().__init__()
        self.lstm_head = nn.LSTM(embedding_dim, lstm_hidden_dim, bidirectional=True)
        self.lstm_tail = nn.LSTM(embedding_dim, lstm_hidden_dim, bidirectional=True)
        self.lstm_src_text = nn.LSTM(embedding_dim, lstm_hidden_dim, bidirectional=True)
        self.fnn = nn.Sequential(
            nn.Dropout(drop_out),
            nn.Linear(5 * 2 * lstm_hidden_dim, ffn_hidden_dim),  # Bidirectional * 2
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(ffn_hidden_dim, class_num)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.class_num = class_num

    def forward(self, head_embed, tail_embed, src_embed, labels=None):
        h_e1, _ = self.lstm_head(head_embed)
        h_e1 = h_e1[:, 0, :]
        h_e2, _ = self.lstm_tail(tail_embed)
        h_e2 = h_e2[:, 0, :]
        h_s, _ = self.lstm_src_text(src_embed)
        h_s = h_s[:, 0, :]
        feature = torch.cat((h_e1, h_e2, h_e1 - h_e2, h_e1 * h_e2, h_s), dim=1)
        logits = self.fnn(feature)
        if labels is not None:
            loss = self.criterion(logits.view(-1, self.class_num), labels.view(-1))
            return loss, logits

        return logits


def train(model, data_loader, saved_model_dir, lr, n_epochs):
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    process_bar = tqdm(range(n_epochs * len(data_loader)))
    for epoch in range(n_epochs):
        for data in data_loader:
            data = {k: v.cuda() for k, v in data.items()}
            loss = model(**data)[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            process_bar.update(1)
            process_bar.set_description('loss=%5.3f' % loss.item())
    torch.save(model.state_dict(), saved_model_dir)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Paths of the training tuples",
                        default='./data/deco/deco_train.json')
    parser.add_argument("--lr", type=str, default=1e-3, help='Learning rate')
    parser.add_argument("--batch_size", type=int, default=32, help='Batch size')
    parser.add_argument("--epoch", type=int, default=10, help='Number of training epochs')
    parser.add_argument("--max_sequence_length", type=int, default=32,
                        help='Max sequence length for truncation and padding.')
    parser.add_argument("--model_dir", type=str, help='Saved model dir.')
    parser.add_argument("--num_proc", type=int, default=1, help="Number of workers for data processing.")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    glove_embed = GloVe(name='840B', dim=300)

    def preprocess_function(examples):
        model_inputs = {
            'head_embed': torch.zeros(len(examples['head']), max_sequence_length, embed_dim),
            'tail_embed': torch.zeros(len(examples['tail']), max_sequence_length, embed_dim),
            'src_embed': torch.zeros(len(examples['src_text']), max_sequence_length, embed_dim),
            'labels': examples['label']

        }
        for i in range(len(examples['head'])):
            head_tok = tokenizer(examples['head'][i])
            head_tok = head_tok + [""] * (max_sequence_length - len(head_tok)) if len(
                head_tok) < max_sequence_length else head_tok[:max_sequence_length]
            tail_tok = tokenizer(examples['tail'][i])
            tail_tok = tail_tok + [""] * (max_sequence_length - len(tail_tok)) if len(
                tail_tok) < max_sequence_length else tail_tok[:max_sequence_length]
            src_tok = tokenizer(examples['src_text'][i])
            src_tok = src_tok + [""] * (max_sequence_length - len(src_tok)) if len(
                src_tok) < max_sequence_length else src_tok[:max_sequence_length]
            model_inputs['head_embed'][i] = glove_embed.get_vecs_by_tokens(head_tok)
            model_inputs['tail_embed'][i] = glove_embed.get_vecs_by_tokens(tail_tok)
            model_inputs['src_embed'][i] = glove_embed.get_vecs_by_tokens(src_tok)
        model_inputs['head_embed'] = model_inputs['head_embed'].tolist()
        model_inputs['tail_embed'] = model_inputs['tail_embed'].tolist()
        model_inputs['src_embed'] = model_inputs['src_embed'].tolist()

        return model_inputs

    tokenizer = get_tokenizer('basic_english')
    max_sequence_length = args.max_sequence_length
    embed_dim = 300

    for rel in RELATIONS:
        positive_dataset = {
            'head': [],
            'tail': [],
            'src_text': [],
            'label': [],
        }
        negative_dataset = {
            'head': [],
            'tail': [],
            'src_text': [],
            'label': [],
        }
        data = load_json(args.data_dir)
        # Process data.
        for sample in tqdm(data):
            utt = sample['response']
            prev_utt = sample['history'].split('</UTT>')[-2]
            for t in sample['tuples_single']:
                if t[1] == rel:
                    positive_dataset['head'].append(t[0])
                    positive_dataset['tail'].append(t[2])
                    positive_dataset['src_text'].append(utt)
                    positive_dataset['label'].append(1)
                else:
                    negative_dataset['head'].append(t[0])
                    negative_dataset['tail'].append(t[2])
                    negative_dataset['src_text'].append(utt)
                    negative_dataset['label'].append(0)
            for t in sample['tuples_pair']:
                if t[1] == rel:
                    positive_dataset['head'].append(t[0])
                    positive_dataset['tail'].append(t[2])
                    positive_dataset['src_text'].append(' '.join([prev_utt, utt]))
                    positive_dataset['label'].append(1)
                else:
                    negative_dataset['head'].append(t[0])
                    negative_dataset['tail'].append(t[2])
                    negative_dataset['src_text'].append(' '.join([prev_utt, utt]))
                    negative_dataset['label'].append(1)
        positive_dataset = Dataset.from_dict(positive_dataset)
        negative_dataset = Dataset.from_dict(negative_dataset)
        # Down sample the negative data.
        negative_dataset = negative_dataset.shuffle(seed=2022)
        negative_dataset = negative_dataset.select(range(len(positive_dataset)))
        train_dataset = concatenate_datasets([positive_dataset, negative_dataset])
        column_names = train_dataset.column_names
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=args.num_proc,
            remove_columns=column_names,
            desc="Preprocessing the train dataset",
        )

        data_collator = default_data_collator
        data_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=True)
        model = LSTMRelationPredictor()
        model = model.cuda()

        train(model, data_loader, os.path.join(args.model_dir, f'{rel}.pth'), args.lr, args.epoch)


if __name__ == '__main__':
    main()
