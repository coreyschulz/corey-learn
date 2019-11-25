import numpy as np

import decisionTree as id3
from data import Data



def get_data_obj(filename):
    data = np.loadtxt(filename, delimiter=',', dtype=str)
    return Data(data=data)


def main():

    print("\nFull Decision Tree: ")
    data_obj = get_data_obj('data/data_semeion/hand_data_train.csv')
    id3_tree = id3.id3(data_obj, data_obj.attributes, data_obj.get_column('label'))
    print(id3_tree)
    print('root node: ', id3_tree.getAttributeName())

    error, depth = id3.getOverallError(data_obj, id3_tree)
    print("Error on training data: {}%; Depth: {}".format(error, depth))

    data_obj_test = get_data_obj('data/data_semeion/hand_data_test.csv')

    error, depth = id3.getOverallError(data_obj_test, id3_tree)
    print("    Error on test data: {}%; Depth: {}".format(error, depth))

    print("\nTree with Max Depth 5")

    max_depth = 5
    pruned_tree = id3.limitTreeDepth(id3_tree, max_depth)

    error, depth = id3.getOverallError(data_obj_test, pruned_tree)
    print("    Error on test data: {}%; Depth: {}".format(error, depth))

    # x = id3.group_label(data_obj)
    #
    # for attribute in data_obj.attributes.keys():
    #     y = id3.group_attribute_by_label(data_obj, data_obj.get_column([attribute, 'label']))
    #     print('{} = > {}'.format(attribute, id3.attribute_expected_entropy(x, y)))
    #
    # for attribute in data_obj.attributes.keys():
    #     y = id3.group_attribute_by_label(data_obj, data_obj.get_column([attribute, 'label']))
    #     print('{} = > {}'.format(attribute, id3.gain(x, y)))


main()
