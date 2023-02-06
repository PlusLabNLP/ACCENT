"""Encode the triplet through [CLS], h, [SEP], r, [SEP], t, [SEP]
"""
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple

import datasets
import pandas as pd
import torch
import transformers
from datasets import Dataset
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed, AutoModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, HPSearchBackend

logger = logging.getLogger(__name__)

RELATIONS = [
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

SELECTED_RELATIONS = [
    "xWant", "oWant", "xEffect", "oEffect", "xReact", "oReact", "xAttr", "xIntent", "xNeed", "isAfter",
    "HinderedBy", "HasSubEvent"
]


class MyTrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        """We don't save the optimizer to save space."""
        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is not None and trial is not None:
            if self.hp_search_backend == HPSearchBackend.OPTUNA:
                run_id = trial.number
            elif self.hp_search_backend == HPSearchBackend.SIGOPT:
                run_id = trial.id
            elif self.hp_search_backend == HPSearchBackend.WANDB:
                import wandb

                run_id = wandb.run.id
            run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
            run_dir = os.path.join(self.args.output_dir, run_name)
        else:
            run_dir = self.args.output_dir
            self.store_flos()

        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)


class KGBertForPopulation(nn.Module):
    """
    (h, r, t) is encoded by feeding '[CLS] h [SEP] r [SEP] t' to the encoder and the embedding of [CLS] is fed into
    a binary classifier.
    """

    def __init__(self, model_name: str):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        self.head = nn.Linear(self.encoder.config.hidden_size, 2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.head(sequence_output[:, 0, :])  # Take <s> / [CLS] token embedding.

        loss = None
        if labels is not None:
            loss = self.criterion(logits.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: int = field(
        default=64,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    preprocessing_num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    encoder_name: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the saved checkpoint."}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup log_to_console
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.encoder_name,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = KGBertForPopulation(
        model_args.encoder_name
    )

    if model_args.checkpoint_dir is not None:
        model.load_state_dict(torch.load(model_args.checkpoint_dir, map_location='cpu'))

    input_column = 'input'
    label_column = 'label'
    raw_datasets = {}
    # Prepare the dataset.
    if training_args.do_train:
        df = pd.read_csv(data_args.train_file)
        df = df.dropna()
        train_dataset = {
            input_column: [],
            label_column: []
        }
        for i in tqdm(range(len(df))):
            head = df.iloc[i]['head']
            tail = df.iloc[i]['tail']
            rel = df.iloc[i]['relation']
            train_dataset[input_column].append(
                ' '.join([head, tokenizer.sep_token, rel, tokenizer.sep_token, tail, tokenizer.sep_token]))
            train_dataset[label_column].append(df.iloc[i]['label'])
        raw_datasets['train'] = Dataset.from_dict(train_dataset)
    if training_args.do_eval:
        validation_dataset = {
            input_column: [],
            label_column: []
        }
        eval_file_dir = './data/evaluation_set.csv'
        df = pd.read_csv(eval_file_dir)
        for i in tqdm(range(len(df))):
            if df.iloc[i]['split'] == 'dev':
                head = df.iloc[i]['head']
                tail = df.iloc[i]['tail']
                rel = df.iloc[i]['relation']
                validation_dataset[input_column].append(
                    ' '.join([head, tokenizer.sep_token, rel, tokenizer.sep_token, tail, tokenizer.sep_token])
                )
                validation_dataset[label_column].append(df.iloc[i]['majority_vote'])
        raw_datasets['validation'] = Dataset.from_dict(validation_dataset)
    if training_args.do_predict:
        test_dataset = {
            input_column: [],
            label_column: []
        }
        test_rels = []
        eval_file_dir = './data/evaluation_set.csv'
        df = pd.read_csv(eval_file_dir)
        for i in tqdm(range(len(df))):
            if df.iloc[i]['split'] == 'tst':
                head = df.iloc[i]['head']
                tail = df.iloc[i]['tail']
                rel = df.iloc[i]['relation']
                test_dataset[input_column].append(
                    ' '.join([head, tokenizer.sep_token, rel, tokenizer.sep_token, tail, tokenizer.sep_token])
                )
                test_dataset[label_column].append(df.iloc[i]['majority_vote'])
                test_rels.append(rel)
        raw_datasets['test'] = Dataset.from_dict(test_dataset)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(examples[input_column], padding=padding, max_length=max_seq_length, truncation=True)
        result["labels"] = examples[label_column]

        return result

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        with training_args.main_process_first(desc="dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        with training_args.main_process_first(desc="dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        with training_args.main_process_first(desc="dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(ignore_keys_for_eval=["encoder_last_hidden_state"])
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Testing
    if training_args.do_eval:
        outputs = trainer.predict(eval_dataset)
        auto_score = softmax(outputs[0], axis=1)[:, 1]
        roc_auc = roc_auc_score(y_true=outputs[1], y_score=auto_score)
        print(f'Valid roc_auc: {roc_auc}')
    if training_args.do_predict:
        outputs = trainer.predict(predict_dataset)
        auto_score = softmax(outputs[0], axis=1)[:, 1]
        roc_auc = roc_auc_score(y_true=outputs[1], y_score=auto_score)
        print(f'Test roc_auc: {roc_auc}')
        # Selected relations
        selected_labels = []
        selected_scores = []
        for i in range(len(auto_score)):
            if test_rels[i] in SELECTED_RELATIONS:
                selected_scores.append(auto_score[i])
                selected_labels.append(outputs[1][i])
        roc_auc_selected_relations = roc_auc_score(y_true=selected_labels, y_score=selected_scores)
        print(f'Test roc_auc (selected relations): {roc_auc_selected_relations}')
        # Breakdown results
        breakdown_results = []
        for rel in RELATIONS:
            tmp_scores = []
            tmp_labels = []
            for i in range(len(auto_score)):
                if test_rels[i] == rel:
                    tmp_scores.append(auto_score[i])
                    tmp_labels.append(outputs[1][i])
            breakdown_results.append(roc_auc_score(y_true=tmp_labels, y_score=tmp_scores))
        print(*RELATIONS, sep='\t')
        print(*breakdown_results, sep='\t')


if __name__ == "__main__":
    main()
