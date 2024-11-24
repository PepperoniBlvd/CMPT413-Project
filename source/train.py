import argparse

import evaluate
import numpy as np
import peft
import torch
import transformers
from datasets import load_dataset
from peft import LoftQConfig, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NLITrainer:

    def __init__(self, model_name, virtual_tokens, prefix_projection, epochs, batch_size, lr, training_percent,
                 finetune_method):
        if any(k in model_name for k in ("gpt", "opt", "bloom")):
            padding_side = "left"
        else:
            padding_side = "right"

        self.model_name = model_name
        self.virtual_tokens = virtual_tokens
        self.prefix_projection = prefix_projection
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        if "prefix" in finetune_method.lower():
            self.finetune = "prefix"
        elif "lora" in finetune_method.lower():
            self.finetune = "lora"
        else:
            raise AssertionError(
                f"Finetune method {finetune_method} is not supported. We currently have 'prefix' and 'lora'.")

        self.metric = evaluate.load("glue", "mnli")
        self.train_percent = training_percent
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)

        if getattr(self.tokenizer, "pad_token_id") is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.dataset = self.load_dataset()
        self.model = self.load_model()

    def load_dataset(self):
        # Use the SNLI dataset
        dataset = load_dataset("snli")
        dataset = dataset.filter(lambda x: x["label"] != -1)

        # select a subset of the training data to train
        indices = np.random.default_rng().choice(len(dataset["train"]),
                                                 size=int(len(dataset["train"]) * self.train_percent), replace=False)
        dataset["train"] = dataset["train"].select(indices)

        # tokenize the dataset so it works with the model input
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True, remove_columns=["premise", "hypothesis"])

        # The label column name used in the model is called "labels"
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        return tokenized_datasets

    def load_model(self):
        if self.finetune == "prefix":
            # prefix tuning: set the hyperparameters
            model = transformers.AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=3).to(
                device)
            model.config.pad_token_id = self.tokenizer.pad_token_id
            prefix_tuning_config = peft.PrefixTuningConfig(peft_type="PREFIX_TUNING", task_type="SEQ_CLS",
                                                           num_virtual_tokens=self.virtual_tokens,
                                                           prefix_projection=self.prefix_projection,
                                                           inference_mode=False)
            model = peft.get_peft_model(model, prefix_tuning_config)
        elif self.finetune == "lora":
            # LoRA: Also add the quantization needed
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                                     bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            model = transformers.AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=3,
                                                                                    quantization_config=quantization_config)  # no to since bitsan
            model = prepare_model_for_kbit_training(model)
            model.config.pad_token_id = self.tokenizer.pad_token_id
            loftq_config = LoftQConfig(loftq_bits=4)
            lora_config = peft.LoraConfig(task_type="SEQ_CLS", r=2, lora_alpha=2, lora_dropout=0.1,
                                          loftq_config=loftq_config)  # can modify as hyperparams
            model = peft.get_peft_model(model, lora_config)
        else:
            raise AssertionError(f"Finetune type must be either 'prefix' or 'lora'!")
        model.print_trainable_parameters()
        model = model.to(device)
        return model

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        # predictions is originally a tensor of logits for the class probabilities
        predictions = np.argmax(predictions, axis=1)
        # self.metric should be accuracy
        return self.metric.compute(predictions=predictions, references=labels)

    def tokenize_function(self, examples):
        # Tokenize the input (premise and hypothesis)
        # Use a max length of 150 tokens and pad if needed
        outputs = self.tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=150,
                                 padding=True, return_tensors="pt").to(device)
        return outputs

    def train(self):
        # Pad the data
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding="longest")
        training_args = TrainingArguments(output_dir="output_models/gpt2-snli-finetune", disable_tqdm=False,
                                          learning_rate=self.lr, per_device_train_batch_size=self.batch_size,
                                          per_device_eval_batch_size=self.batch_size, num_train_epochs=self.epochs,
                                          weight_decay=0.01, evaluation_strategy="epoch", save_strategy="epoch",
                                          logging_strategy="steps", log_level="info", load_best_model_at_end=True,
                                          bf16=True)
        # Training is simple with HuggingFace's Trainer class, just need to set training arguments
        trainer = Trainer(model=self.model, args=training_args, train_dataset=self.dataset["train"],
                          eval_dataset=self.dataset["validation"], tokenizer=self.tokenizer,
                          data_collator=data_collator, compute_metrics=self.compute_metrics)

        trainer.train()


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-M", "--basemodel", dest="basemodel", default='distilgpt2',
                           help="The base huggingface pretrained model to be used as the encoder.")
    argparser.add_argument("-v", "--virtualtokens", dest="virtualtokens", type=bool, default=5,
                           help="number of virtual prompt tokens for prefix tuning")
    argparser.add_argument("-p", "--prefixprojection", dest="prefixprojection", action="store_true", default=False,
                           help="whether to project the prefix embeddings")
    argparser.add_argument("-e", "--epochs", dest="epochs", type=int, default=1, help="number of epochs [default: 1]")
    argparser.add_argument("-b", "--batchsize", dest="batchsize", type=int, default=16, help="batch size [default: 16]")
    argparser.add_argument("-r", "--lr", dest="lr", type=float, default=5e-5,
                           help="the learning rate used to finetune the BERT-like encoder module.")
    argparser.add_argument("-n", "--trainingpercent", dest="trainingpercent", type=float, default=1.0,
                           help="the percent of training examples to use to train the model.")
    argparser.add_argument("-f", "--finetune", dest="finetune", type=str, default="prefix",
                           help="The type of finetune method. 'prefix' or 'lora'")
    argparser.add_argument("-c", "--checkpoint", dest="checkpoint", type=str, default=None,
                           help="The checkpoint folder to use to resume training.")
    opts = argparser.parse_args()
    trainer = NLITrainer(opts.basemodel, opts.virtualtokens, opts.prefixprojection, opts.epochs, opts.batchsize,
                         opts.lr, opts.trainingpercent, opts.finetune)
    trainer.train()


if __name__ == '__main__':
    main()
