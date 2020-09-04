import os


# Get subfolders; names
def subfolder_names(path):
    items_list = []
    for r, d, files in os.walk(path):
        if len(r.split("\\")) > 1:
            items_list.append(r.split("\\")[1])
    return items_list


def acc_classes(path, acc_index):
    folders_list = subfolder_names(path)

    result_list = []


    for (i,x) in enumerate(acc_index):
        if x[0] > 0.001:
            result_list.append({
                "accuracy": x[0],
                "class": folders_list[x[1]]
            })

    return result_list