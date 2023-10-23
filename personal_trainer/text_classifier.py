import json
import numpy as np
import math
import os
import torch

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging, TrainingArguments, Trainer

from .label_mapper import LabelMapper

logging.set_verbosity_error()

class BaseTextClassifier:
    def __init__(self, fine_tuned_path=None):
        self.label_mapper = LabelMapper()
        self.tokenizer = None
        self.model = None

        if fine_tuned_path is not None:
            self.model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_path)
            self.tokenizer = AutoTokenizer.from_pretrained(fine_tuned_path)
            with open(os.path.join(fine_tuned_path, 'personal_trainer.json')) as f:
                info = json.load(f)
                self.label_mapper = LabelMapper(info['labels'])

    def _train(self, texts, labels, multi_label):
        if self.model is not None:
            raise RuntimeError("'train' can only be called on untrained models")
        elif len(texts) == 0:
            raise ValueError('dataset must include at least one value to train on')

        model_name = 'thenlper/gte-base'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        hf_dataset = Dataset.from_dict({'text': texts, 'labels': labels})
        def tokenize_function(datapoints):
            return self.tokenizer(datapoints["text"], padding="max_length", truncation=True)
        tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1)

        problem_type = "single_label_classification"
        def count_correct(logits, labels):
            predictions = np.argmax(logits, axis=-1)
            return (predictions == labels).astype(int).sum()

        if multi_label:
            problem_type = "multi_label_classification"

            def count_correct(logits, labels):
                probabilities = 1.0 / (1.0 + np.exp(-logits))
                predictions = (probabilities > 0.95).astype(int)
                correct = 0
                for i in range(len(predictions)):
                    if (predictions[i] == labels[i]).all():
                        correct += 1
                return correct

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.label_mapper.num_labels(),
            problem_type=problem_type
        )

        # Since we have a randomly initiated classifier head, we need to make a minimum number of
        # parameter updates to have a well performing model. Because of this we pick our number of
        # epochs based on how many parameter updates we want to make, that's been chosen heuristically
        # based on the default learning rate
        target_total_batches = 350
        batch_size = 8
        batches_per_epoch = math.ceil(len(split_dataset['train']) / batch_size)
        num_epochs = math.ceil(target_total_batches / batches_per_epoch)

        training_args = TrainingArguments(
            output_dir='/tmp/personal_trainer',
            evaluation_strategy="epoch",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            correct = count_correct(logits, labels)
            accuracy = float(correct) / len(logits)
            return {'accuracy': accuracy}

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=split_dataset['train'],
            eval_dataset=split_dataset['test'],
            compute_metrics=compute_metrics
        )

        self.trainer.train()

    def save(self, path):
        if self.trainer is None:
            raise RuntimeError("'save' can only be called after training")
        self.trainer.save_model(path)
        self.tokenizer.save_pretrained(path)

        personal_trainer_path = os.path.join(path, 'personal_trainer.json')
        with open(personal_trainer_path, 'w+') as f:
            json.dump({'labels': self.label_mapper.index_to_label}, f)

    def _classify(self, text):
        if self.model is None:
            raise RuntimeError("'classify' can't be called before either fine-tuning or loading a fine-tuned model")
        input = self.tokenizer([text], truncation=True, return_tensors='pt')
        input = input.to(self.model.device)
        res = self.model(**input)
        return res.logits.cpu().detach().numpy()

def is_simple_value(val):
    return isinstance(val, str) or isinstance(val, int) or isinstance(val, bool) or val is None

class TextClassifier(BaseTextClassifier):
    def __init__(self, fine_tuned_path=None):
        super().__init__(fine_tuned_path)

    def train(self, dataset):
        texts = []
        labels = []

        for (text, label) in dataset:
            if not is_simple_value(label):
                raise TypeError("label '{}' is not a string, integer, boolean, or None".format(label))
            self.label_mapper.maybe_add_label(label)
            label_idx = self.label_mapper.index_for_label(label)
            texts.append(text)
            labels.append(label_idx)

        super()._train(texts, labels, False)

    def classify(self, text):
        logits = super()._classify(text)
        predictions = np.argmax(logits, axis=-1)
        return self.label_mapper.label_for_index(predictions[0])

class MultiLabelTextClassifier(BaseTextClassifier):
    def __init__(self, fine_tuned_path=None):
        super().__init__(fine_tuned_path)

    def train(self, dataset):
        texts = []
        label_tensors = []

        for (text, labels) in dataset:
            texts.append(text)
            for label in labels:
                if not isinstance(label, str) and not isinstance(label, int):
                    raise TypeError(f"label '{label}' is not a string or integer")

                self.label_mapper.maybe_add_label(label)

        num_labels = self.label_mapper.num_labels()
        for (text, labels) in dataset:
            label_bits = [1 if self.label_mapper.label_for_index(i) in labels else 0 for i in range(num_labels)]
            label_tensor = torch.tensor(label_bits, dtype=torch.float)
            label_tensors.append(label_tensor)

        super()._train(texts, label_tensors, True)

    def classify(self, text, threshold=0.95):
        logits = super()._classify(text)
        probabilities = 1.0 / (1.0 + np.exp(-logits))

        results = []
        for i in range(self.label_mapper.num_labels()):
            if probabilities[0][i] > threshold:
                results.append(self.label_mapper.label_for_index(i))

        return results
