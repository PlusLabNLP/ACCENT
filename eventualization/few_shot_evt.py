"""Train the event-relation extraction model through prompt-based few-shot learning."""
import logging
import sys
from typing import Optional

import os
import pandas as pd
import torch
import transformers
from dataclasses import dataclass, field
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer import PREFIX_CHECKPOINT_DIR
from transformers.trainer_utils import get_last_checkpoint, HPSearchBackend

from utils import load_json

logger = logging.getLogger(__name__)

# 12 Selected Relations.
RELATIONS = ['xNeed', 'xAttr', 'xReact', 'xEffect', 'xWant', 'xIntent', 'oEffect', 'oReact', 'oWant',
             'isAfter', 'HasSubEvent', 'HinderedBy']


class MySeq2SeqTrainer(Seq2SeqTrainer):
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


class Eventualization:
    def __init__(self, with_context, one_previous_utterance_only):
        self.xNeed_prompt = "Extract event1 and event2 from the text where event2 needs to be true for event1 to take place. Utterance: "
        self.xAttr_prompt = "Extract event1 and event2 from the text where event2 shows how PersonX is viewed as after event1. Utterance: "
        self.xReact_prompt = "Extract event1 and event2 from the text where event2 shows how PersonX reacts to event1. Utterance: "
        self.xEffect_prompt = "Extract event1 and event2 from the text where event2 shows the effect of event1 on PersonX. Utterance: "
        self.xWant_prompt = "Extract event1 and event2 from the text where event2 shows what PersonX wants after event1 happens. Utterance: "
        self.xIntent_prompt = "Extract event1 and event2 from the text where event2 shows PersonX's intent for event1. Utterance: "
        self.oEffect_prompt = "Extract event1 and event2 from the text where event2 shows the effect of event1 on PersonY. Utterance: "
        self.oReact_prompt = "Extract event1 and event2 from the text where event2 shows how PersonY reacts to event1. Utterance: "
        self.oWant_prompt = "Extract event1 and event2 from the text where event2 shows what PersonY wants after event1 happens. Utterance: "
        self.IsAfter_prompt = "Extract event1 and event2 from the text where event1 happens after event2. Utterance: "
        self.subEvent_prompt = "Extract event1 and event2 from the text where event1 includes event2. Utterance: "
        self.HinderedBy_prompt = "Extract event1 and event2 from the text where event1 fails to happen because event2. Utterance: "

        self.prompts = [self.xNeed_prompt, self.xAttr_prompt, self.xReact_prompt, self.xEffect_prompt,
                        self.xWant_prompt, self.xIntent_prompt, self.oEffect_prompt, self.oReact_prompt,
                        self.oWant_prompt, self.IsAfter_prompt, self.subEvent_prompt, self.HinderedBy_prompt]
        self.relations = RELATIONS

        self.with_context = with_context
        self.one_previous_utterance_only = one_previous_utterance_only

    def load_train_data(self, input_file, none_sample_file, include_single=True, include_pair=True):
        all_data = load_json(input_file)
        self.data = {}
        for rel in self.relations:
            self.data[rel] = []
        for sample in all_data:
            context = sample['history']
            response = sample['response']
            if include_single:
                for t in sample['tuples_single']:
                    # Represent the tuple in a pre-defined format.
                    self.data[t[1]].append((context, response, f'event1: {t[0].strip()}; event2: {t[2].strip()}'))
            if include_pair:
                for t in sample['tuples_pair']:
                    # Represent the tuple in a pre-defined format.
                    self.data[t[1]].append((context, response, f'event1: {t[0].strip()}; event2: {t[2].strip()}'))
        # Load negative samples.
        negative_samples = pd.read_csv(none_sample_file)
        for i in range(len(negative_samples)):
            if not include_single and negative_samples.at[i, 'setting'] == 'single':
                continue
            if not include_pair and negative_samples.at[i, 'setting'] == 'pair':
                continue
            rel = negative_samples.at[i, 'relation']
            context = negative_samples.at[i, 'context']
            response = negative_samples.at[i, 'response']
            self.data[rel].append((context, response, 'None'))

    def process_data(self):
        """Attach prompt to the input text to construct the training pairs."""
        src = []
        tgt = []
        rels = []
        for ind_rel, rel in enumerate(self.relations):
            for data in self.data[rel]:
                try:
                    context = data[0].split('</UTT>')[-2] + '</UTT>' if self.one_previous_utterance_only else data[0]
                    input_sent = context + data[1] if self.with_context else data[1]
                    input_sent = self.prompts[ind_rel] + input_sent
                    output_sent = data[2]
                    src.append(input_sent)
                    tgt.append(output_sent)
                    rels.append(rel)
                except Exception as e:
                    print(e)
                    import pdb
                    pdb.set_trace()

        return src, tgt, rels

    def inference(self, sample, model, tokenizer, max_length):
        """Do eventualization on sample.

        sample: (previous context, current utterance)
        """
        model.eval()
        evt_results = {}
        input_sentences = []
        corresponding_rels = []
        with torch.no_grad():
            for idx, prompt in enumerate(self.prompts):
                if self.relations[idx] not in RELATIONS:
                    continue  # We only care about our interested relations.
                test_sent = prompt + sample[0] + sample[1] if self.with_context else prompt + sample[1]
                input_sentences.append(test_sent)
                corresponding_rels.append(self.relations[idx])

            inputs = tokenizer(input_sentences, padding=True, return_tensors='pt').to(model.device)

            beam_outputs = model.generate(
                **inputs,
                max_length=max_length,
                early_stopping=True,
                num_beams=10,
                num_return_sequences=1,
                no_repeat_ngram_size=2
            )
            for beam_output, rel in zip(beam_outputs, corresponding_rels):
                sent = tokenizer.decode(beam_output, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True)
                evt_results[rel] = sent

        return evt_results


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
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
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    none_sample_file: Optional[str] = field(
        default=None, metadata={"help": "The input negative samples file (csv)."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metric (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
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
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    with_context: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether include dialogue context in model input or not"
        }
    )
    include_none: bool = field(
        default=True,
        metadata={
            "help": "Whether to include samples without target relation."
        }
    )
    one_previous_utterance_only: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only consider the one previous utterance."
        }
    )
    include_single: bool = field(
        default=True,
        metadata={
            "help": "Whether to include tuples in Single setting"
        }
    )
    include_pair: bool = field(
        default=True,
        metadata={
            "help": "Whether to include tuples in Pair setting"
        }
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
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
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Eventualization setup.
    evt = Eventualization(data_args.with_context, data_args.one_previous_utterance_only)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    # Load the data.
    text_column = 'src'
    output_column = 'tgt'
    column_names = [text_column, output_column]
    if training_args.do_train:
        evt.load_train_data(data_args.train_file, data_args.none_sample_file,
                            include_pair=data_args.include_pair, include_single=data_args.include_single)
        src, tgt, rels = evt.process_data()
        train_dataset = {
            text_column: src,
            output_column: tgt
        }
        train_dataset = Dataset.from_dict(train_dataset)

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(examples):
        # remove pairs where at least one record is None
        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] is not None and examples[output_column][i] is not None:
                inputs.append(examples[text_column][i])
                targets.append(examples[output_column][i])

        inputs = [inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Initialize our Trainer
    trainer = MySeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        compute_metrics=None,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()


if __name__ == "__main__":
    main()
