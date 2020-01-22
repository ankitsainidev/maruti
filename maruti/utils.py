import json


def read_json(path):
    '''
    Read Json file as dict.
    '''
    with open(path,'rb') as file:
        json_dict = json.load(file)
    return json_dict

def write_json(dictionary, path):
    """
    Write dict as a json file
    """
    with open(path,'w') as fp:
        json.dump(dictionary, fp)