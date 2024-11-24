import argparse

import datasets
import torch
import transformers
from datasets import load_dataset, Dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NLITester:

    def __init__(self, model_name, checkpoint, file_name, out_file_name, batch_size):
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        if any(k in model_name for k in ("gpt", "opt", "bloom")):
            padding_side = "left"
        else:
            padding_side = "right"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
        if getattr(self.tokenizer, "pad_token_id") is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = self.load_model(model_name, checkpoint)
        self.dataset = self.load_dataset(file_name)
        self.file_name = file_name
        self.out_file_name = out_file_name

    def load_dataset(self, file_name):
        has_label = True
        if file_name is None:
            # No file name given, use SNLI test set
            dataset = load_dataset("snli", split=datasets.Split.TEST)
            dataset = dataset.filter(lambda x: x["label"] != -1)
        else:
            dataset = {"premise": [], "hypothesis": [], "label": []}
            with open(file_name, "r") as f:
                # assuming split char is "|" and first row is header
                sc = "|"
                has_label = "label" in f.readline().lower()
                # rest of lines
                for line in f.readlines():
                    toks = line.split(sc)
                    dataset["premise"].append(toks[0])
                    dataset["hypothesis"].append(toks[1])
                    if has_label:
                        dataset["label"].append(int(toks[2]))  # label
                    else:
                        dataset["label"].append(-1)  # will be dropped later
            dataset = Dataset.from_dict(dataset)

        tokenized_datasets = dataset.map(self.tokenize_function, batched=True, remove_columns=["premise", "hypothesis"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        if not has_label:
            tokenized_datasets = tokenized_datasets.remove_columns("labels")
        return tokenized_datasets

    def load_model(self, model_name, checkpoint):
        # Need to set pad token
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        model.config.pad_token_id = self.tokenizer.pad_token_id
        # Load in the trained PEFT weights from the checkpoint folder
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.to(device)
        return model

    def tokenize_function(self, examples):
        # Tokenize the input (premise and hypothesis)
        # Use a max length of 150 tokens and pad if needed
        outputs = self.tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=150,
                                 padding=True, return_tensors="pt").to(device)
        return outputs

    def evaluate(self):
        write_to_file = self.out_file_name is not None
        # Number of correct predictions
        correct = torch.tensor(0, device=device)
        # Set the data needed
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding="longest")
        eval_dataloader = DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=data_collator)
        # List of predictions to write to a file (if output file is given)
        ret = []
        total_len = 0
        # Evaluation loop
        for i, d in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            f = d.to(device)
            output = self.model(**f)
            logits = output.logits
            argmax = torch.argmax(logits, dim=1).float()
            # If the labels are given, compute the test accuracy
            if "labels" in f:
                correct += torch.sum(f["labels"] == argmax)
            total_len += len(argmax)
            if write_to_file:
                ret += list(argmax.int().cpu().numpy())

        if self.out_file_name is not None:
            print(f"Writing results to {self.out_file_name}...")
            lab_map = {
                0: "Entailment", 1: "Neutral", 2: "Contradiction"
            }
            # Map the label ids to words
            with open(self.out_file_name, "w+") as f:
                f.writelines([lab_map[i] + "\n" for i in ret])

        # Acc will be 0 if no labels provided
        return (correct / total_len).item()


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-M", "--basemodel", dest="basemodel", default='gpt2', required=True,
                           help="The base huggingface pretrained model to be used as the encoder.")
    argparser.add_argument("-b", "--batchsize", dest="batchsize", type=int, default=16, help="batch size [default: 16]")
    argparser.add_argument("-c", "--checkpoint", dest="checkpoint", type=str, default=None, required=True,
                           help="The checkpoint folder containing the model to evaluate.")
    argparser.add_argument("-f", "--file", dest="file", default=None,
                           help="The input file to test with. If not provided, the SNLI test set will be downloaded and used.")
    argparser.add_argument("-o", "--outfile", dest="outfile", default=None,
                           help="The output file to write eval results to. Only works when --file is provided.")
    opts = argparser.parse_args()

    tester = NLITester(opts.basemodel, opts.checkpoint, opts.file, opts.outfile, opts.batchsize)
    if opts.file is None:
        print(f"Evaluation accuracy on the SNLI test dataset: {tester.evaluate()}")
    else:
        e = tester.evaluate()
        print(f"Evaluation accuracy on {opts.file}: {'N/A' if e == 0 else e}.")


if __name__ == '__main__':
    main()
