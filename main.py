import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
import scipy
import pandas as pd

from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop,Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
import warnings

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
warnings.filterwarnings('ignore')
np.random.seed(2)
dict_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson',
        3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel',
        7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lenny_leonard', 11:'lisa_simpson',
        12: 'marge_simpson', 13: 'mayor_quimby',14:'milhouse_van_houten', 15: 'moe_szyslak',
        16: 'ned_flanders', 17: 'nelson_muntz', 18: 'principal_skinner', 19: 'sideshow_bob'}
# Load the data
def load_train_set(dirname,dict_characters):
    X_train = []
    Y_train = []
    for label,character in dict_characters.items():
        list_images = os.listdir(dirname+'/'+character)
        for image_name in list_images:
            image = scipy.misc.imread(dirname+'/'+character+'/'+image_name)
            X_train.append(scipy.misc.imresize(image,(64,64),interp='lanczos'))
            Y_train.append(label)
    return np.array(X_train), np.array(Y_train)


# load the test data

def load_test_set(dirname,dict_characters):
    X_test = []
    Y_test = []
    for image_name in os.listdir(dirname):
        character_name = "_".join(image_name.split('_')[:-1])
        label = [label for label,character in dict_characters.items() if character == character_name][0]
        image = scipy.misc.imread(dirname+'/'+image_name)
        X_test.append(scipy.misc.imresize(image,(64,64),interp='lanczos'))
        Y_test.append(label)
    return np.array(X_test), np.array(Y_test)
def load_ml_test(dirname):
    X_test = []
    for image_name in os.listdir(dirname):
        image = scipy.misc.imread(dirname+'/'+image_name)
        X_test.append(scipy.misc.imresize(image,(64,64),interp='lanczos'))
    return np.array(X_test)

X_train, Y_train = load_train_set("C:/Users/CaesarYu/Desktop/ML_HW02/data/train", dict_characters)
X_train,X_val, Y_train,Y_val =train_test_split(X_train, Y_train, test_size = 0.33 , random_state = 42)
X_val_out, Y_val_out = load_test_set("C:/Users/CaesarYu/Desktop/ML_HW02/data/kaggle_simpson_testset", dict_characters)
X_test=load_ml_test("C:/Users/CaesarYu/Desktop/ML_HW02/data/test")
# Normalize the data
X_test=X_test/255.0
X_train = X_train / 255.0
X_val = X_val / 255.0
X_val_out = X_val_out / 255.0
# 0~1


def display_samples(samples_index, imgs, obs, preds_classes, preds):
    """This function randomly displays 20 images with their observed labels
    and their predicted ones(if preds_classes and preds are provided)"""
    n = 0
    nrows = 4
    ncols = 5
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(12, 10))
    plt.subplots_adjust(wspace=0, hspace=0)
    for row in range(nrows):
        for col in range(ncols):
            index = samples_index[n]
            ax[row, col].imshow(imgs[index])

            actual_label = dict_characters[obs[index]].split("_")[0]
            actual_text = "Actual : {}".format(actual_label)

            ax[row, col].add_patch(patches.Rectangle((0, 53), 64, 25, color='white'))
            font0 = FontProperties()
            font = font0.copy()
            font.set_family("fantasy")

            ax[row, col].text(1, 54, actual_text, horizontalalignment='left', fontproperties=font,
                              verticalalignment='top', fontsize=10, color='black', fontweight='bold')

            if preds_classes != 'none' and preds != 'none':
                predicted_label = dict_characters[preds_classes[index]].split('_')[0]
                predicted_proba = max(preds[index]) * 100
                predicted_text = "{} : {:.0f}%".format(predicted_label, predicted_proba)

                ax[row, col].text(1, 59, predicted_text, horizontalalignment='left', fontproperties=font,
                                  verticalalignment='top', fontsize=10, color='black', fontweight='bold')
            n += 1


def pick_up_random_element(elem_type, array):
    """This function randomly picks up one element per type in the array"""
    return int(random.choice(np.argwhere(array == elem_type)))


samples = [pick_up_random_element(elem_type, Y_train) for elem_type in range(20)]

display_samples(samples, X_train, Y_train,'none','none')

# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 20)
Y_val = to_categorical(Y_val, num_classes = 20)
Y_val_out = to_categorical(Y_val_out, num_classes = 20)

# Set the CNN model
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',
                 activation ='relu', input_shape = (64,64,3)))
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

# ↑ Features Extraction
model.add(Flatten())
# 全連接層(?)
#model.add(Dense(1024, activation = "relu"))
#model.add(Dropout(0.5))
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(20, activation = "softmax"))

# Define the optimizer
#optimizer = RMSprop(lr=0.001, decay=1e-6)
optimizer = Adadelta(decay=1e-6)

PRETRAINED = False  #True

if PRETRAINED:
    json_file = open('best_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("best_model.hdf5")

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
# Set a learning_rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)


save_best = ModelCheckpoint("best_model.hdf5", monitor='val_acc', verbose=0, save_best_only=True, mode='max')
#把訓練中的最好的模型存起來

if PRETRAINED == False:
    epochs = 60 #總共學幾次(一次就是全部資料學光光)
    batch_size = 320 #多少個一起丟

    # With data augmentation to prevent the overfitting (accuracy 0.97)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(X_train)

    # Fit the model
    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
                                  epochs=epochs, validation_data=(X_val, Y_val),
                                  steps_per_epoch=X_train.shape[0] // batch_size
                                  , callbacks=[learning_rate_reduction, save_best],verbose=2)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    plt.savefig('history')
    # serialize model to JSON
    model_json = model.to_json()
    with open("best_model.json", "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        # model.save_weights("model_big.h5")

# predict results
loss, acc = model.evaluate(X_val, Y_val, verbose = 0)
loss1, acc1 = model.evaluate(X_val_out, Y_val_out, verbose = 0)
print("ML:Simpson characters were predicted with a loss of {:.5f} and an accuracy of {:.2f}%".format(loss,acc*100))
print("OTHERS:Simpson characters were predicted with a loss of {:.5f} and an accuracy of {:.2f}%".format(loss1,acc1*100))

# Look at confusion matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert hot vectors prediction results to list of classes
Y_pred_classes = np.argmax(Y_pred, axis=1)
# Convert hot vectors validation observations to list of classes
Y_true = np.argmax(Y_val, axis=1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes=list(dict_characters.values()))
Y_PREDICTION = model.predict(X_test)
Y_PREDICTION_classes = np.argmax(Y_PREDICTION, axis=1)
result_name=[]
for i in Y_PREDICTION_classes:
     result_name.append(dict_characters[i])
print(result_name)
result=pd.DataFrame(result_name)
result.index=np.arange(1,len(result_name)+1)
result.index.name='id'
result.columns=['character']
result.to_csv('OUTPUT1.csv')
# Display some results

samples = [pick_up_random_element(elem_type,Y_true) for elem_type in range(20)]
display_samples(samples, X_val, Y_true, Y_pred_classes, Y_pred)
plt.show()