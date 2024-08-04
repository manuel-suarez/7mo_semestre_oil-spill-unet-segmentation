import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt

HOME_DIR = os.path.expanduser('~')
DATA_DIR = os.path.join(HOME_DIR, "data", "oil-spill-dataset_256")

# load repo with data if it is not exists
x_train_dir = os.path.join(DATA_DIR, 'train', 'images')
y_train_dir = os.path.join(DATA_DIR, 'train', 'labels_1D')

x_valid_dir = os.path.join(DATA_DIR, 'val', 'images')
y_valid_dir = os.path.join(DATA_DIR, 'val', 'labels_1D')

x_test_dir = os.path.join(DATA_DIR, 'test', 'images')
y_test_dir = os.path.join(DATA_DIR, 'test', 'labels_1D')

# helper function for data visualization
def visualize(name, image, mask):
    """PLot images in one row."""
    n = 2
    plt.figure(figsize=(16, 5))
    
    plt.subplot(1, n, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title('image')
    plt.imshow(image, cmap='gray')
    
    plt.subplot(1, n, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title('mask')
    plt.imshow(mask, cmap='gray', vmin=0, vmax=4)
    
    # 
    # plt.show()
    plt.savefig(name)

# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['sea', 'oil-spill', 'look-alike', 'ship', 'land']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id.split('.')[0] + '.jpg') for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.split('.')[0] + '.png') for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        print(self.class_values)

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i], 0)
        image = np.expand_dims(image, axis=-1)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = np.expand_dims(mask, axis=-1)
        # print(i, image.shape, mask.shape, np.unique(mask))

        # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')
        # print(i, image.shape, mask.shape)

        # add background if mask is not binary
        # if mask.shape[-1] != 1:
        #     background = 1 - mask.sum(axis=-1, keepdims=True)
        #     mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        # if self.augmentation:
        #     sample = self.augmentation(image=image, mask=mask)
        #     image, mask = sample['image'], sample['mask']

        # apply preprocessing
        # if self.preprocessing:
        #     sample = self.preprocessing(image=image, mask=mask)
        #     image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)

class DataLoader(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

# Lets look at data we have
dataset = Dataset(x_train_dir, y_train_dir, classes=['sea', 'oil-spill', 'look-alike', 'ship', 'land'])
image, mask = dataset[4] # get some sample
print(image.shape)
print(mask.shape)
visualize(
    'figures/figure01.png',
    image=image,
    mask=mask
)

# os.environ['SM_FRAMEWORK'] = 'tf.keras'
# import segmentation_models as sm
# BACKBONE = 'resnet34'
BATCH_SIZE = 32
LR = 0.0001
EPOCHS = 40

# preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
n_classes = 5 
activation = 'softmax'

from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.optimizers import Adam
os.makedirs('results', exist_ok=True)
checkpoint = ModelCheckpoint(os.path.join('results', f"unetv2_checkpoint_set_epochs{EPOCHS}.keras"), monitor='val_accuracy', verbose=0, save_best_only=True, mode='auto')
logger = CSVLogger(os.path.join('results', f"unetv2_training_set_epochs{EPOCHS}.log"))
callbacks = [logger,checkpoint]
    # Optimizer
optimizer = Adam(learning_rate = LR)
#create model
#from models.unet_model_v1 import multi_unet_model
from models.unet_model_v2 import build_unet
#model = sm.Unet(BACKBONE, input_shape=(None, None, 1), classes=n_classes, activation=activation, encoder_weights=None)
#model = multi_unet_model(n_classes=n_classes)
model = build_unet(input_shape=(256,256,1), num_classes=5, activation='softmax')

import tensorflow as tf
class UpdatedMeanIoU(keras.metrics.MeanIoU):
  def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               name=None,
               dtype=None):
    super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    return super().update_state(y_true, y_pred, sample_weight)

import tensorflow as tf
class UpdatedIoU(keras.metrics.IoU):
  def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               target_class_ids=None,
               name=None,
               dtype=None):
    super(UpdatedIoU, self).__init__(num_classes=num_classes, name=name, target_class_ids=target_class_ids, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    return super().update_state(y_true, y_pred, sample_weight)

from keras.metrics import MeanIoU, IoU
from keras.losses import SparseCategoricalCrossentropy

meaniou = UpdatedMeanIoU(num_classes=5, name='mean_iou')
iou_c0 = UpdatedIoU(num_classes=5, target_class_ids=[0], name='iou_class0')
iou_c1 = UpdatedIoU(num_classes=5, target_class_ids=[1], name='iou_class1')
iou_c2 = UpdatedIoU(num_classes=5, target_class_ids=[2], name='iou_class2')
iou_c3 = UpdatedIoU(num_classes=5, target_class_ids=[3], name='iou_class3')
iou_c4 = UpdatedIoU(num_classes=5, target_class_ids=[4], name='iou_class4')
loss = SparseCategoricalCrossentropy(from_logits=False)
#metrics = ['accuracy','precision','recall',meaniou,iou_c0,iou_c1]
metrics = ['accuracy',meaniou,iou_c0,iou_c1,iou_c2,iou_c3,iou_c4]
# Compile with loss function
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.summary()

# Dataset for train images
train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    classes=[]
)

# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    classes=[]
)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE//4, shuffle=False)

# check shapes for errors
assert train_dataloader[0][0].shape == (BATCH_SIZE, 256, 256, 1)
assert train_dataloader[0][1].shape == (BATCH_SIZE, 256, 256, 1)

# Train
model.fit(train_dataloader,
          validation_data = valid_dataloader,
          epochs = EPOCHS,
          callbacks = callbacks,
          verbose = 1)

test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    classes=[]
)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE//8, shuffle=False)
model.evaluate(test_dataloader)
