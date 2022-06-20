import numpy as np
from tensorflow import keras
import json


def load_labels():
    CLASS_INDEX_PATH = ('https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json')
    fpath = keras.utils.get_file(
        'imagenet_class_index.json',
        CLASS_INDEX_PATH,
        cache_subdir='models',
        file_hash='c2c37ea517e94d9795004a39431a14cb')
    with open(fpath) as f:
        CLASS_INDEX = json.load(f)
    label_imnet = [CLASS_INDEX[str(i)][1] for i in range(len(CLASS_INDEX))]
    label_cifar10 = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    label_cifar100 = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    return (np.array(label_imnet), np.array(label_cifar10), np.array(label_cifar100))


def quantile_norm( x, ref_dist ):
    # for x
    if len(x.shape)==2:
        n_column = x.shape[1]
        N = x.shape[0]
    else:
        n_column = 1
        x = x.reshape([-1,n_column])
        N = x.shape[0]
    # for reference distribution
    if ref_dist.shape[0]==N:
        if len(ref_dist.shape)==1:
            y = np.sort(ref_dist)
        elif len(ref_dist.shape)==2:
            y = np.sort(ref_dist,axis=0).mean(axis=1)
        else:
            y = np.quantile( ref_dist.flatten(), np.arange(0,1,1/N)+1/(2*N) )
    else:
        y = np.quantile( ref_dist.flatten(), np.arange(0,1,1/N)+1/(2*N) )
    # replace values
    x2 = x.copy()
    for i in range( n_column ):
        iarg = np.argsort(x[:,i])
        x2[iarg,i] = y
    return x2


def get_x( model, add_images_pp, add_labels ):
    insize = model.input_shape[1]
    w = model.layers[-1].get_weights()
    Nhide = w[0].shape[0]
    Nclas = w[1].shape[0]
    hidden_model = keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    add_id = np.sort(np.unique(add_labels))
    nadd = len(add_id)
    for ic in range(nadd):
        xhide = np.zeros( Nhide )
        ids = np.where( add_labels == add_id[ic] )[0]
        for ik in range( len(ids) ):
            x = add_images_pp[ids[ik],:,:,:]
            x = np.expand_dims(x, 0)
            buf = hidden_model.predict(x).reshape(-1)
            xhide += buf
        xhide = xhide/len(ids)
        if ic==0:
            x_add = xhide.copy().reshape([-1,1])
        else:
            x_add = np.hstack( [ x_add, xhide.reshape([-1,1]) ] )
    return x_add


def change_w(model, x_add, reconstruct):
    wori = model.layers[-1].get_weights()
    Nclas = wori[1].shape[0]
    hidden_model = keras.Model(inputs=model.input, outputs=model.layers[-2].output)

    Nadd = x_add.shape[1]
    w0add = quantile_norm( x_add, wori[0] )
    w1add = np.zeros(Nadd) + np.median(wori[1])

    wadd = [[]]*2
    if reconstruct:
        wadd[0] = w0add
        wadd[1] = w1add
        Nclas_new = Nadd
    else:
        wadd[0] = np.hstack( [wori[0], w0add] )
        wadd[1] = np.hstack( [wori[1], w1add] )
        Nclas_new = Nclas + Nadd

    top_layer = keras.layers.Dense(  Nclas_new , activation='softmax' )( hidden_model.output )
    model_add = keras.Model( inputs=hidden_model.input, outputs=top_layer )
    model_add.layers[-1].set_weights(  wadd  )

    return model_add



def add_class( model, add_images_pp, add_labels=0, reconstruct=0):
    if len(add_labels)==0: add_labels=np.zeros(len(add_images_pp))
    x_add = get_x( model, add_images_pp, add_labels )
    model_add = change_w( model, x_add, reconstruct )
    return model_add


def attach_resize(model, data_shape):
    model_inputs = keras.Input(shape=data_shape)
    x = keras.layers.experimental.preprocessing.Resizing(model.input_shape[1],model.input_shape[2])(model_inputs)
    model_outputs = model(x)
    model = keras.Model(inputs=model_inputs, outputs=model_outputs)
    return model
