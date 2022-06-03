# tfdone

TensorFlow implementation of DONE ([It's DONE: Direct ONE-shot learning with Hebbian weight imprinting](https://arxiv.org/abs/2204.13361)). DONE is the simplest one-shot learning method that can add new classes to a pretrained DNN without optimization or the backbone-DNN modification.


![scheme of DONE](https://raw.githubusercontent.com/hosodakazufumi/tfdone/master/fig/fig1.png)



## Requirements

* `numpy` and  `tensorflow`
> I have checked `numpy==1.20.3`, `tensorflow==2.9.1` or `tensorflow-gpu==2.7.0` work.


## Installation

```bash
$ pip install tfdone
```


## Usage

* To add new classes to a model with some training data and training label (see below, e.g., you can obtain a 1003-class model if you add 3 classes to a ImageNet 1000-class model):

```python
from tfdone import done
model_added = done.add_class( model, data, label)
```
> - *data.shape = (num_images, height, width, channels)*  
> - *label.shape = (num_images,)*  
> - *model: must have a flattened Dense layer at the top. If not, use Keras to arrange it. In most cases, this can be solved by flattening, removing layers after Dense layer, or unpacking a multi-layer complex.* 


* To refresh classes (transfer learning), just put `reconstruct=1`:

```python
model_new = done.add_class( model, data, label, reconstruct=1)
```



## Examples  (see `done_example.ipynb` file)

This example shows a case of adding new three classes to a 1000-class ImageNet model of EfficientNet-B0 using 1, 2, and 3 images of the new classes 'baby', 'caterpillar', and 'sunflower', respectively (i.e., 1-shot learning for 'baby', and 2- and 3-shot learning for 'caterpillar' and 'sunflower', respectively). See [done_example.ipynb](https://github.com/hosodakazufumi/tfdone/blob/main/done_example.ipynb) file.
> I recommend using Vision Transformer (ViT) as a backbone model for DONE, but for simplicity I here use EfficientNet, which is included in TensorFlow. When using ViT, for example, I have confirmed [vit-keras](https://github.com/faustomorales/vit-keras) works. 

```python
import numpy as np
import tensorflow as tf
from tfdone import done

# backbone model (e.g., EfficientNet; I recommend using ViT though)
model = tf.keras.applications.efficientnet.EfficientNetB0()

# Image data (e.g., from CIFAR-100)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

# Let's say we have 1, 2, 3 images of baby, caterpillar, sunflower. 
images = x_train[[202,75,89,12,164,274]].copy() # Images
add_y = np.array([0,1,1,2,2,2]) # Labels

# Resize & preprocess training images
images_resized = tf.image.resize(images, model.input_shape[1:3])
images_processed = tf.keras.applications.efficientnet.preprocess_input(images_resized)

# Class addition by DONE
model_added = done.add_class( model, images_processed, add_y)

# It's DONE. You obtained a 1003-class model.

```


## Other useful functions  (see `done_example.ipynb` file)
```python
# Quantile normalization
x_new = done.quantile_norm( x, reference )

# Attaching input resize layer
model_resize = done.attach_resize(model, (height, width, channels))

# Labels of ImageNet, CIFAR-10, and CIFAR-100
(label_imnet, label_cifar10, label_cifar100) = done.load_labels()
```



## Acknowledgements
I would like to thank all the contributors to relevant open source software such as TensorFlow. 

## References

1) **Original weight imprinting**: Hang Qi, Matthew Brown, David G. Lowe. "Low-Shot Learning with Imprinted Weights", in CVPR, 2018.
2)  **DONE**: Kazufumi Hosoda, Keigo Nishida, Shigeto Seno, Tomohiro Mashita, Hideki Kashioka, Izumi Ohzawa. "It's DONE: Direct ONE-shot learning with Hebbian weight imprinting", arXiv, 2204.13361.

