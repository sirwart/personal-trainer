import json
import numpy as np
import os

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging, TrainingArguments, Trainer

logging.set_verbosity_error()

def is_simple_value(val):
    return isinstance(val, str) or isinstance(val, int) or isinstance(val, bool) or val is None

class TextClassifier:
    def __init__(self, fine_tuned_path=None):
        self.label_to_index = None
        self.index_to_label = []
        self.tokenizer = None
        self.model = None

        if fine_tuned_path is not None:
            self.model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_path)
            self.tokenizer = AutoTokenizer.from_pretrained(fine_tuned_path)
            with open(os.path.join(fine_tuned_path, 'personal_trainer.json')) as f:
                info = json.load(f)
                self.index_to_label = info['labels']
                self.label_to_index = {}
                for i in range(len(self.index_to_label)):
                    label = self.index_to_label[i]
                    self.label_to_index[label] = i

    def train(self, dataset):
        if self.label_to_index is not None:
            raise RuntimeError("'train' can only be called on untrained models")
        elif len(dataset) == 0:
            raise ValueError('dataset must include at least one value to train on')

        self.label_to_index = {}

        model_name = 'bert-base-cased'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        texts = []
        labels = []

        for (text, label) in dataset:
            if not is_simple_value(label):
                raise TypeError("label '{}' is not a string, integer, or boolean".format(label))
            if label not in self.label_to_index:
                label_idx = len(self.label_to_index)
                self.label_to_index[label] = label_idx
                self.index_to_label.append(label)
            else:
                label_idx = self.label_to_index[label]
            texts.append(text)
            labels.append(label_idx)
        
        hf_dataset = Dataset.from_dict({'text': texts, 'label': labels})
        def tokenize_function(datapoints):
            return self.tokenizer(datapoints["text"], padding="max_length", truncation=True)
        tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1)

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(self.label_to_index))
        training_args = TrainingArguments(output_dir='/tmp/personal_trainer', evaluation_strategy="epoch")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            correct = (predictions == labels).astype(int).sum()
            accuracy = float(correct) / len(predictions)
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
            json.dump({'labels': self.index_to_label}, f)

    def classify(self, text):
        if self.model is None:
            raise RuntimeError("'classify' can't be called before either fine-tuning or loading a fine-tuned model")
        input = self.tokenizer([text], truncation=True, return_tensors='pt')
        input = input.to(self.model.device)
        res = self.model(**input)
        predictions = np.argmax(res.logits.cpu().detach().numpy(), axis=-1)
        return self.index_to_label[predictions[0]]
