import json


def open_json(path):
    '''
    Read Json file as dict.
    '''
    with open(path) as file:
        json_dict = json.load(file)
    return json_dict
