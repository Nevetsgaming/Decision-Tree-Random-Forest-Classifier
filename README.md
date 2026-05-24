# Decision-Tree-Random-Forest-Classifier
My attempt at exercises 7-8 from Hands on Machinelearning with Pytorch chapter 5
# Random Forest Ensemble Experiment

This project explores how random forests improve on single decision trees using Scikit-Learn.

The goal was to understand ensemble learning by first training a single decision tree, then training many decision trees on random subsets of the data, and finally combining their predictions with majority voting.

## Main Idea

A single decision tree can overfit because it is highly sensitive to the training data. If the data changes slightly, the tree structure can change a lot.

A random forest reduces this instability by training many different decision trees and combining their predictions. Each tree may make mistakes, but if the trees are different enough, majority voting usually gives a more stable final prediction.

## What I Did

- Loaded a classification dataset
- Trained a single Scikit-Learn decision tree
- Used `ShuffleSplit` to create many random training subsets
- Trained many decision trees on these subsets
- Combined the trees using majority voting
- Compared the custom ensemble to Scikit-Learn's `RandomForestClassifier`


## Results

The single decision tree performed worse than the ensemble in most trials. The ensemble was more stable because it averaged out the mistakes of individual trees.
