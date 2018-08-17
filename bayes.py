#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

from collections import defaultdict

from scipy.io import arff


class BayesClassifier(object):
    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file

    @staticmethod
    def parse_arff(filename):
        return arff.loadarff(filename)

    @staticmethod
    def parse_meta_data(meta):
        d = {}
        class_attrs = None
        for name in meta.names():
            if name == "class":
                class_attrs = meta[name][1]
                continue
            d[name] = meta[name][1]
        return class_attrs, d

    @staticmethod
    def class_probabilities(data, class_attrs):
        d = {}
        d_len = float(len(data))
        for attr in class_attrs:
            fdlen = len(filter(lambda record: record["class"] == attr, data))
            d[attr] = (fdlen + 1) / (d_len + len(class_attrs))
        return d

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError


class NaiveBayesClassifier(BayesClassifier):
    def __init__(self, train_file, test_file):
        super(NaiveBayesClassifier, self).__init__(train_file, test_file)

    def __str__(self):
        return "Naive Bayes"

    @staticmethod
    def rvar_probabilities(data, meta, class_attrs):
        filtered_data = {}
        for cattr in class_attrs:
            filtered_data[cattr] = \
                filter(lambda record: record["class"] == cattr, data)

        rvar_probs = {}
        for attr_name, attr_values in meta.iteritems():
            rvar_probs[attr_name] = defaultdict(dict)
            for attr_value in attr_values:
                rvar_probs[attr_name][attr_value] = defaultdict(dict)
                for cattr, fdata in filtered_data.iteritems():
                    fdlen = float(len(fdata))
                    fd = filter(lambda record: record[attr_name] == attr_value,
                                fdata)
                    rvar_probs[attr_name][attr_value][cattr] = (len(fd) + 1) / \
                                                               (fdlen + len(attr_values))

        return rvar_probs

    def train(self, data=None, meta=None):
        if data is None or meta is None:
            data, meta = BayesClassifier.parse_arff(self.train_file)

        class_attrs, meta = BayesClassifier.parse_meta_data(meta)
        if class_attrs is None:
            raise Exception("meta data does not contain class attributes")

        self.class_probs = BayesClassifier.class_probabilities(data, class_attrs)
        self.rvar_probs = NaiveBayesClassifier.rvar_probabilities(data, meta,
                                                                  class_attrs)

    def test(self, data=None, meta=None):
        if data is None or meta is None:
            data, meta = BayesClassifier.parse_arff(self.test_file)

        for name in meta.names():
            if name == "class":
                continue

            print "%s class" % name

        print

        correctly_classified = 0
        classification_probs = []

        for record in data:
            probs = {}
            for cattr, cprob in self.class_probs.iteritems():
                probs[cattr] = cprob
                for name in meta.names():
                    if name == "class":
                        continue
                    probs[cattr] *= self.rvar_probs[name][record[name]][cattr]

            denom = sum(probs.values())
            for cattr in self.class_probs.keys():
                probs[cattr] /= denom

            predicted_class = max(probs, key=probs.get)
            if predicted_class == record["class"]:
                correctly_classified += 1

            print "%s %s %.12f" % (predicted_class, record["class"],
                                probs[predicted_class])

            classification_probs.append((predicted_class, record["class"], probs))

        print "\n%s" % correctly_classified
        return classification_probs


class TanBayesClassifier(BayesClassifier):
    def __init__(self, train_file, test_file):
        super(TanBayesClassifier, self).__init__(train_file, test_file)

    def __str__(self):
        return "TAN Bayes"

    @staticmethod
    def calculate_mutual_information(data, class_attrs, meta):
        zipped_names = [(name1, name2) for name1 in meta.keys()
                        for name2 in meta.keys()]

        information = defaultdict(int)

        for name1, name2 in zipped_names:
            info = 0
            if name1 == name2:
                information[(name1, name2)] = -1.0
                continue

            for attr1 in meta[name1]:
                for attr2 in meta[name2]:
                    for cattr in class_attrs:
                        fd1 = float(
                            len(
                                filter(
                                    lambda rec: rec[name1] == attr1 and
                                                rec[name2] == attr2 and
                                                rec["class"] == cattr, data)
                            )
                        )

                        fd2 = filter(lambda rec: rec["class"] == cattr, data)

                        fd3 = float(len(filter(
                            lambda rec: rec[name1] == attr1 and
                                        rec[name2] == attr2, fd2)))

                        fd4 = float(len(filter(
                            lambda rec: rec[name1] == attr1, fd2)))

                        fd5 = float(len(filter(
                            lambda rec: rec[name2] == attr2, fd2)))

                        fd2len = float(len(fd2))

                        pfd1 = (fd1 + 1) / (len(data) + (len(meta[name1]) *
                                                         len(meta[name2]) *
                                                         len(class_attrs)))

                        pfd3 = (fd3 + 1) / (fd2len + (len(meta[name1]) *
                                                      len(meta[name2])))

                        pfd4 = (fd4 + 1) / (fd2len + len(meta[name1]))
                        pfd5 = (fd5 + 1) / (fd2len + len(meta[name2]))

                        info += pfd1 * math.log((pfd3 / (pfd4 * pfd5)), 2)

            information[(name1, name2)] = info

        return information

    def build_graph(self, info, meta, precedence):
        queue = set(name for name in meta.keys() if name != precedence[0])

        self.graph = defaultdict(list)
        chosen_vertices = set()
        chosen_vertices.add(precedence[0])

        while queue:
            possible_edges = {}
            for src in chosen_vertices:
                for dst in queue:
                    possible_edges[(src, dst)] = info[(src, dst)]

            maxwedge = max(possible_edges.values())
            contented_edges = [edge for edge in possible_edges.iterkeys()
                               if possible_edges[edge] == maxwedge]

            chosen_edge = sorted(contented_edges,
                                 key=lambda x: (precedence.index(x[0]),
                                                precedence.index(x[1])))[0]

            src, dst = chosen_edge
            queue.remove(dst)
            chosen_vertices.add(dst)
            self.graph[src].append(dst)

    def get_parent(self, attr):
        for src, dsts in self.graph.iteritems():
            if attr in dsts:
                return src
        return None

    def build_cond_probs(self, data, meta):
        self.cond_probs = {}

        for attr_name, attr_values in meta.iteritems():
            parent = self.get_parent(attr_name)
            for attr_val in attr_values:
                for cattr, cprob in self.class_probs.iteritems():
                    if parent is not None:
                        for pattr_val in meta[parent]:
                            d = filter(lambda rec: rec["class"] == cattr and
                                                   rec[parent] == pattr_val, data)

                            dlen = float(len(d))
                            fd = filter(lambda rec: rec[attr_name] == attr_val, d)
                            comb = ("%s=%s" % (attr_name, attr_val),
                                    "%s=%s" % (parent, pattr_val),
                                    "class=%s" % cattr)
                            self.cond_probs[comb] = (len(fd) + 1) / (dlen + len(attr_values))
                    else:
                        d = filter(lambda rec: rec["class"] == cattr, data)
                        dlen = float(len(d))
                        fd = filter(lambda rec: rec[attr_name] == attr_val, d)
                        comb = ("%s=%s" % (attr_name, attr_val),
                                "%s=%s" % (attr_name, attr_val),
                                "class=%s" % cattr)
                        self.cond_probs[comb] = (len(fd) + 1) / (dlen + (len(attr_values)))

    def printtree(self, names):
        for name in names:
            if name == "class":
                continue

            parent = self.get_parent(name)
            ustr = name
            if parent:
                ustr += " " + parent
            ustr += " class"
            print ustr
        print

    def train(self, data=None, meta=None):
        if data is None or meta is None:
            data, meta = BayesClassifier.parse_arff(self.train_file)

        precedence = meta.names()
        class_attrs, meta = BayesClassifier.parse_meta_data(meta)
        if class_attrs is None:
            raise Exception("meta data does not contain class attributes")
        info = TanBayesClassifier.calculate_mutual_information(data,
                                                               class_attrs,
                                                               meta)

        self.class_probs = BayesClassifier.class_probabilities(data,
                                                               class_attrs)
        self.root_attr = precedence[0]
        self.build_graph(info, meta, precedence)
        self.build_cond_probs(data, meta)
        self.printtree(precedence)

    def test(self, data=None, meta=None):
        # P(C | X1, …, X𝑛) = P(C) ∙ P(Xroot | C) ∏P(Xi|C, X𝑝𝑎𝑟𝑒𝑛𝑡)
        if data is None or meta is None:
            data, meta = BayesClassifier.parse_arff(self.test_file)

        correctly_classified = 0
        classification_probs = []

        for record in data:
            probs = {}
            rval = record[self.root_attr]
            for cattr, cprob in self.class_probs.iteritems():
                comb = ("%s=%s" % (self.root_attr, rval),
                        "%s=%s" % (self.root_attr, rval),
                        "class=%s" % cattr)

                probs[cattr] = cprob * self.cond_probs[comb]
                for name in meta.names():
                    if name in ["class", self.root_attr]:
                        continue

                    aval = record[name]
                    parent = self.get_parent(name)
                    pval = record[parent]
                    comb = ("%s=%s" % (name, aval),
                            "%s=%s" % (parent, pval),
                            "class=%s" % cattr)
                    probs[cattr] *= self.cond_probs[comb]

            denom = sum(probs.values())
            for cattr in self.class_probs.keys():
                probs[cattr] /= denom

            predicted_class = max(probs, key=probs.get)
            if predicted_class == record["class"]:
                correctly_classified += 1

            print "%s %s %.12f" % (predicted_class, record["class"],
                                   probs[predicted_class])

            classification_probs.append((predicted_class, record["class"], probs))

        print "\n%s" % correctly_classified
        return classification_probs


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print "Usage: python bayes.py <train-set-file> <test-set-file> <n|t>"

    classifiers = {
        'n': NaiveBayesClassifier,
        't': TanBayesClassifier
    }

    train_file, test_file, ctype = sys.argv[1:]
    classifier = classifiers[ctype](train_file, test_file)
    classifier.train()
    classifier.test()
