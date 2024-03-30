import os.path

"""
Class for representing fixed label categories

properties
- connective: list of connective area labels, str list

create instance
    label_list()
"""
class label_list():
    """
    create labels object

    parameters
    - file_path: location of connective labels file, str
        (default '../data/connective.txt')

    returns
        label_list object
    """
    def __init__(self, file_path=(os.path.dirname(__file__) + "/../data/connective.txt")):
        file = open(file_path, "r")
        lines = file.readlines()

        self.connective = [line.strip() for line in lines]
        self.all = self.connective

    """
    create string of labels object

    invoked by print(), str()
    """
    def __repr__(self):
        return str(self.connective)