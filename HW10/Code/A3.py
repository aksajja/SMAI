import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

train1 = unpickle('./cifar-10-batches-py/data_batch_1')

for key,value in train1.items():
    # b'batch_label'
    # b'labels'
    # b'data'
    # b'filenames'
    if key==b'data':
        print(value.shape)
