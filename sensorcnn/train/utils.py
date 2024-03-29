import os
import errno
import csv
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix


def get_bytes_from_file(filename):
    """Read a file into a byte array"""
    return open(filename, "rb").read()


def remove_if_exists(filename):
    """Makes sure a file at a location is writable.
    Checks if file at the location exists. Deletes it if it is there and ensures all parent directories are present."""
    if os.path.exists(filename):
        os.remove(filename)
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(os.path.dirname(filename)):
            pass
        else:
            raise


def write_to_csv(filename, data):
    """Write csv data to filename"""
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


def evaluate(y_true, y_pred, labels):
    num_labels = len(labels)

    confusion_mat = confusion_matrix(y_true, y_pred, range(0, num_labels))

    # Fiddle around with cm to get it into table shape
    confusion_mat = np.vstack((np.zeros((1, num_labels), dtype=int), confusion_mat))
    confusion_mat = np.hstack((np.zeros((num_labels + 1, 1), dtype=int), confusion_mat))

    table = confusion_mat.tolist()

    for i, s in enumerate(labels):
        table[0][i+1] = s
        table[i+1][0] = s

    table[0][0] = "actual \ predicted"
    return table


def timestamp():
    return int(datetime.today().timestamp())


def export(model, evaluation_tabel, setup_labels):
    root_path = os.path.join("output", model.name, str(timestamp()))
    os.makedirs(root_path)
    write_to_csv(os.path.join(root_path, "evaluation_table.csv"), evaluation_tabel)
    with open(os.path.join(root_path, "labels.txt"), 'w') as f:
        f.writelines("\n".join(setup_labels))

    # Export the model
    model.save_weights(os.path.join(root_path, "weights.raw"))
    model.save_layers(os.path.join(root_path, "layers.txt"))
    print("Model saved to %s" % root_path)
