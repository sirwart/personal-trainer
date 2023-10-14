import os
import sys
sys.path.append(os.getcwd())

from personal_trainer import MultiLabelTextClassifier

dataset = [
        ('impala', ['car', 'animal']),
        ('viper', ['car', 'animal']),
        ('mustang', ['car', 'animal']),
        ('ram', ['car', 'animal']),
        ('bronco', ['car', 'animal']),
        ('beetle', ['car', 'animal']),
        ('jaguar', ['car', 'animal']),
        ('escape', ['car']),
        ('fiesta', ['car']),
        ('rubicon', ['car']),
        ('el camino', ['car']),
        ('malibu', ['car']),
        ('sedona', ['car']),
        ('tacoma', ['car']),
        ('hot dog', []),
        ('tree', []),
        ('smartphone', []),
        ('jeans', []),
        ('speaker', []),
        ('government', []),
        ('chair', [])
]

classifier = MultiLabelTextClassifier()

classifier.train(dataset)

classifier.save('/tmp/classifier')

classifier = MultiLabelTextClassifier('/tmp/classifier')

input = 'cayman'

res = classifier.classify(input)

print('classify({}):'.format(input), res)

if res != ['car', 'animal']:
    raise ValueError('Not what was expected')




