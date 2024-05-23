import numpy as np


def calc_accuracy(predictions: [[float]], true_labels: [int], val):
    tp = tn = fp = fn = 0

    for output, true in zip(predictions, true_labels):
        predicted = np.argmax(output)
        if predicted == val and true == val:
            tp += 1
        elif predicted != val and true != val:
            tn += 1
        elif predicted == val and true != val:
            fp += 1
        elif predicted != val and true == val:
            fn += 1

    return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0


def calc_precision(predictions: [[float]], true_labels: [int], val):
    true_positive = 0
    false_positive = 0

    for i in range(len(predictions)):
        predicted_label = np.argmax(predictions[i])
        if predicted_label == true_labels[i] == val:
            true_positive += 1
        elif predicted_label == val or true_labels[i] == val:
            false_positive += 1

    return true_positive / (true_positive + false_positive)


def calc_recall(predictions: [[float]], true_labels: [int], val):
    true_positive = 0
    false_negative = 0

    for i in range(len(predictions)):
        predicted_label = np.argmax(predictions[i])
        if predicted_label == true_labels[i] == val:
            true_positive += 1
        elif true_labels[i] == val:
            false_negative += 1

    return true_positive / (true_positive + false_negative)


def calc_multi_auc(predictions: [[float]], true_labels):
    num_classes = len(predictions[0])
    auc_total = 0

    for class_idx in range(num_classes):
        class_predictions = [pred[class_idx] for pred in predictions]
        auc_class = auc(class_predictions, true_labels, class_idx)
        auc_total += auc_class

    auc_avg = auc_total / num_classes
    return auc_avg


def auc(class_predictions: [float], true_labels, class_idx):
    num_positive = 0
    num_negative = 0
    sum_rank_positive = 0

    for i in range(len(true_labels)):
        if true_labels[i] == class_idx:
            num_positive += 1
            for j in range(len(true_labels)):
                if true_labels[j] != class_idx:
                    if class_predictions[i] > class_predictions[j]:
                        sum_rank_positive += 1
                    elif class_predictions[i] == class_predictions[j]:
                        sum_rank_positive += 0.5
        else:
            num_negative += 1

    auc_class = sum_rank_positive / (num_positive * num_negative)
    return auc_class
