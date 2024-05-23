import pandas as pd

from src.Network import Network
from src.layer.ConvalutionalLayer import ConvalutionalLayer
from src.layer.DenseLayer import DenseLayer
from src.layer.ReshapeLayer import ReshapeLayer
from src.metrics import calc_accuracy, calc_precision, calc_recall, calc_multi_auc
from src.preprocessing import create_datasets, one_hotes_to_digits

NUM_CLASSES = 10


def test_network(network, x_test, y_test):
    predictions = network.predictions(x_test)
    true_labels = one_hotes_to_digits(y_test)

    accuracy = 0
    precision = 0
    recall = 0

    for digit in range(NUM_CLASSES):
        accuracy += calc_accuracy(predictions, true_labels, digit)
        precision += calc_precision(predictions, true_labels, digit)
        recall += calc_recall(predictions, true_labels, digit)

    accuracy /= NUM_CLASSES
    precision /= NUM_CLASSES
    recall /= NUM_CLASSES
    auc = calc_multi_auc(predictions, true_labels)

    print("Accuracy:\t{}".format(accuracy))
    print("Precision:\t{}".format(precision))
    print("Recall:\t{}".format(recall))
    print("Auc:\t\t{}".format(auc))


if __name__ == '__main__':
    df_train = pd.read_csv('resource/mnist_train.csv')
    df_test = pd.read_csv('resource/mnist_test.csv')

    x_train, y_train = create_datasets(df_train)
    x_test, y_test = create_datasets(df_test)

    network = Network([
        ConvalutionalLayer((1, 28, 28), 3, 5),
        ReshapeLayer((5, 26, 26), (5 * 26 * 26, 1)),
        DenseLayer(5 * 26 * 26, 100),
        DenseLayer(100, NUM_CLASSES)
    ])

    network.fit(x_train, y_train, 1, 0.01)

    test_network(network, x_test, y_test)
