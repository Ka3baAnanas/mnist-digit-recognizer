import gzip
import pickle
import numpy as np

def load_data():
    with gzip.open('../data/mnist.pkl.gz',"rb") as f :
        (tr_d,vl_d,test_d)=pickle.load(f , encoding='latin1') #the file was pickled in Python 2 (which Nielsen’s dataset was), then add encoding='latin1'
    return (tr_d,vl_d,test_d)

def load_data_wrapper():
    (tr_d,vl_d,test_d)=load_data()
    images=(tr_d[0],vl_d[0],test_d[0])
    labels= (tr_d[1],vl_d[1],test_d[1])
    data=([],[],[])
    for datatype in range(3):
        for (image, label) in zip(images[datatype], labels[datatype]): 
            
            '''What does zip() do?:
               It pairs elements positionally from multiple iterables:
zip(iterable1, iterable2, ..., iterableN)
It returns an iterator of tuples, where each tuple contains the i-th element from each iterable.

Examples:
list(zip([1, 2], ['a', 'b']))      # [(1, 'a'), (2, 'b')]
list(zip("AB", [100, 200]))        # [('A', 100), ('B', 200)]
list(zip({'x': 1, 'y': 2}, [9, 8]))# [('x', 9), ('y', 8)]'''
            
            image_reshaped= np.reshape(image, (784, 1))
            # ⚠️: Only training labels are one-hot encoded! 
            # Validation and test labels must stay as integers.  
            if datatype == 0 :#(training data)
                label_vectorised=vectorise(label)
                data[datatype].append((image_reshaped,label_vectorised))
            else :
                digit_label=label
                data[datatype].append((image_reshaped,digit_label))
    return data
def vectorise(a):
    v=np.zeros((10,1))
    v[a]=1
    return v














    
