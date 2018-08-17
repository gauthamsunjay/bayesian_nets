from bayes import NaiveBayesClassifier, TanBayesClassifier

import numpy as np
from scipy.io import arff
from scipy import stats

data, meta = arff.loadarff("lymph_cv.arff")

np.random.shuffle(data)

num_folds = 10
dlen = len(data)
pfoldd = dlen / num_folds

naive_accuracies = []
tan_accuracies = []

for i in range(num_folds):
    if i == num_folds - 1:
        test_data = data[i*pfoldd:]
        train_data = data[:i*pfoldd]
    else:
        test_data = data[i * pfoldd:(i + 1) * pfoldd]
        train_data = np.concatenate(
            (data[:i * pfoldd], data[(i + 1) * pfoldd:]), axis=0)

    classifier = NaiveBayesClassifier("lymph_train.arff", "lymph_test.arff")
    classifier.train(data=train_data, meta=meta)
    num_correct = 0
    for plabel, alabel, _ in classifier.test(data=test_data, meta=meta):
        if plabel == alabel:
            num_correct += 1

    naive_accuracies.append(num_correct / float(len(test_data)))

    classifier = TanBayesClassifier("lymph_train.arff", "lymph_test.arff")
    classifier.train(data=train_data, meta=meta)
    num_correct = 0
    for plabel, alabel, _ in classifier.test(data=test_data, meta=meta):
        if plabel == alabel:
            num_correct += 1

    tan_accuracies.append(num_correct / float(len(test_data)))


deltas = [naive_accuracies[i] - tan_accuracies[i] for i in range(num_folds)]
mean_delta = sum(deltas) / float(len(deltas))
diff_deltas = [(deltas[i] - mean_delta) ** 2 for i in range(num_folds)]

t = mean_delta / ((sum(diff_deltas) / (num_folds * (num_folds - 1))) ** 0.5)
p_value = stats.t.sf(t, num_folds - 1)

print "sample mean = %f" % mean_delta
print "t statistic = %f" % t
print "p Value = %f" % p_value
if p_value < 0.05:
    print "p value is significant. Hence, the alternate hypothesis is true."
