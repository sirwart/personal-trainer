import os
import sys
sys.path.append(os.getcwd())

from personal_trainer import TextClassifier

dataset = [
        ('oscar meyer', 'hot dog'),
        ('choripan', 'hot dog'),
        ('bratwurst', 'hot dog'),
        ('sandwich', 'not'),
        ('burger', 'not'),
        ('dumpling', 'not'),
        ('celery', 'not'),
        ('water', 'not'),
        ('bun', 'not'),
        ('vienna sausage', 'hot dog'),
]

classifier = TextClassifier()

classifier.train(dataset)

classifier.save('/tmp/classifier')

classifier = TextClassifier('/tmp/classifier')

input = 'taco'

res = classifier.classify(input)

print('classify({}):'.format(input), res)

if res != 'hot dog':
    raise ValueError('Not what was expected')




