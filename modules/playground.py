from __future__ import print_function

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # This will hide those Keras messages

"""
    InvceptionV3 has input (299, 299, 3) ((in case the environment is configured to have the channel at the end)
"""


from helper import acc_class

prediction = [(0.55019504, 35), (0.44966635, 65), (0.00013862934, 81), (3.8286053e-08, 31), (3.96442e-15, 33)]

res = acc_class("data/dataset", prediction)

print(res)
