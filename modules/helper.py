import os
import json

# Get subfolders; names
def subfolder_names(path):
    items_list = []
    for r, d, files in os.walk(path):
        if len(r.split("\\")) > 1:
            items_list.append(r.split("\\")[1])
    return items_list


def acc_classes(classesjson_file, acc_index):
    with open(classesjson_file) as json_file:
        classes = json.load(json_file)

    result_list = []

    for (i,x) in enumerate(acc_index):
        if x[0] > 0.001:
            result_list.append({
                "accuracy": float(x[0]),
                "class": classes[x[1]].replace("_", " ")
            })

    return result_list


# Get number of subfolders directly below the folder in path
def get_num_subfolders(path):
    if not os.path.exists(path):
        return 0
    return sum([len(d) for r, d, files in os.walk(path)])

