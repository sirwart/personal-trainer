# Personal Trainer

Personal Trainer is a Python library for easily training models on your own data using open source models.

Personal Trainer is still under development.

# Installation

You can install Personal Trainer using `pip` from the `model-personal-trainer` package.

```
pip install model-personal-trainer
```

# Usage

Each type of model is represented by a class in Personal Trainer. For example, one of the simplest models to use is a TextClassifier.

```
from personal_trainer import TextClassifier

classifier = TextClassifier()
```

A text classifier takes some text and returns whether it's thing A or thing B. To train the model
you first construct a series of examples of text and how it should be classified. You can then pass
those examples to the `train` method of the classifier.

```
examples = [
        ('oscar meyer', 'hot dog'),
        ('choripan', 'hot dog'),
        ('bratwurst', 'hot dog'),
        ('vienna sausage', 'hot dog'),
        ('sandwich', 'not'),
        ('burger', 'not'),
        ('dumpling', 'not'),
        ('bun', 'not'),
]

classifier.train(examples)
```

Once you've trained your model you can then use it. For a classifier that means calling the `classify` method.

```
res = classifier.classify('taco')
print('res:', res)
```

If training was successful `res` should be `not`. A classifier output can be any string, integer, or Boolean.

To save your model for later use using the `save` method. This writes the model out to the path that you
provide.

```
classifier.save('hot-dog-or-not.model')
```

To load the model later, provide the same path when constructing the classifier.

```
classifier = TextClassifier('hot-dog-or-not.model')
```
