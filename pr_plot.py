from scipy.io import arff
import matplotlib.pyplot as plt
from bayes import NaiveBayesClassifier, TanBayesClassifier


def gather_pr(cprobs, testfile):
    data, meta = arff.loadarff(testfile)
    positive, negative = meta["class"][1]

    total_positive = float(len(filter(lambda rec: rec["class"] == positive, data)))
    precision_data = []
    recall_data = []

    cprobs = sorted(cprobs, key=lambda x: x[-1][positive], reverse=True)
    tp = 0
    ppos = 0

    for plabel, alabel, cprob in cprobs:
        if plabel == positive and alabel == positive:
            tp += 1

        if plabel == positive:
            ppos += 1

        precision = tp / float(ppos)
        recall = tp / total_positive

        precision_data.append(precision)
        recall_data.append(recall)

    return precision_data, recall_data


if __name__ == "__main__":
    import sys

    classifiers = {
        'n': NaiveBayesClassifier,
        't': TanBayesClassifier
    }

    if len(sys.argv) < 3:
        print "Usage: python pr_curve.py trainfile testfile"

    trainfile, testfile = sys.argv[1:]
    fig, ax = plt.subplots()

    for c in classifiers:
        classifier = classifiers[c](trainfile, testfile)
        classifier.train()
        cprobs = classifier.test()

        precision_data, recall_data = gather_pr(cprobs, testfile)

        ax.plot(recall_data, precision_data, label="%s" % (classifier, ),
                marker='+')

    ax.set(xlabel="Recall", ylabel="Precision",
           title="PR Curve")

    ax.grid()
    plt.legend()
    fig.savefig("pr_curve.png")
    plt.show()
