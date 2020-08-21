# https://keras.io/applications/

#from keras.applications.inception_v3 import InceptionV3, preprocess_input
import keras
from keras.applications.vgg19 import VGG19, preprocess_input
#from keras.applications.xception import Xception, preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.layers import Flatten
from keras import regularizers
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras import optimizers 
#from keras.utils import multi_gpu_model

#K.set_image_dim_ordering('th')

lr1 = 0.0005
#lr2 = 0.0001
iter1 = 150
#iter2 = 150
n_classes = 11
data_path = "dataset/"
train_dir = data_path + "train/"
test_dir = data_path + 'valid/'

def predict_test(model,train_generator):
	test_dirs = os.listdir(test_dir)
	correct = 0
	count = 0
	for f in test_dirs:
		if os.path.isdir(test_dir+f):
			img_path = glob.glob(test_dir+f+"/"+"*.jpg")
			for i in range(0,len(img_path)):
				img = image.load_img(img_path[i], target_size=(256, 256))
				x = image.img_to_array(img)
				x = np.expand_dims(x, axis=0)
				x = preprocess_input(x)
				preds = model.predict(x)
				ind = np.argmax(preds)
				if ind == train_generator.class_indices[f]:
					correct += 1
				count += 1
	return float(correct)/float(count)

# create the base pre-trained model
base_model = VGG19(weights='imagenet', include_top=False)
#base_model = Xception(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
#x = MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format=None)(x)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
#x = keras.layers.Flatten()(x)
#x = filterReductionLayer()(x)

# let's add a fully-connected layer
x = Dense(2048, activation='relu', kernel_regularizer = regularizers.l2(0.001))(x)
x = Dropout(0.4)(x)

#x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)
x = Dropout(0.3)(x)

x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)


# and a logistic layer -- let's say we have 5 classes
predictions = Dense(n_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
	layer.trainable = False

for layer in model.layers[:18]:
	layer.trainable = False
for layer in model.layers[18:]:
	layer.trainable = True

#model = multi_gpu_model(model, gpus = 4)

# compile the model (should be done *after* setting layers to non-trainable)
#model.compile(optimizers.RMSprop(lr=lr1, decay=0.0), loss='categorical_crossentropy')
model.compile(optimizers.Adam(lr=lr1, decay=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#variable final_model stores name of model file with h5 extension
final_model = "dataset/model/final_model.h5"
final_model_best = "dataset/model/final_model_best.h5"
#final_model_best = load_model('dataset/model/final_model_best.h5')


#train_datagen = image.ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_datagen = image.ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True, rotation_range = 30 )
valid_datagen = image.ImageDataGenerator()#shear_range=0.2, zoom_range=0.2, horizontal_flip=True

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(256, 256),batch_size=100,class_mode='categorical')
valid_generator = valid_datagen.flow_from_directory(test_dir,target_size=(256, 256),batch_size=64,class_mode='categorical')

print (model.summary())


#model checkpoints
model_checkpoint = keras.callbacks.ModelCheckpoint(final_model_best, monitor='val_acc', mode='auto', save_best_only=True)

#early stopping
early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', mode='auto', verbose=1, patience=10)

history = model.fit_generator(train_generator,samples_per_epoch=6000, epochs=iter1, verbose=1, validation_data=valid_generator, validation_steps=64, callbacks=[model_checkpoint, early_stop])

model.save(final_model)

best_model = load_model(final_model_best)


for i, layer in enumerate(base_model.layers):
	print(i, layer.name)

acc_stage1 = predict_test(best_model,train_generator)

print("Accuracy at Stage 1: ",acc_stage1)

# -------------------------------------------
 
#saving graphs for the training and validation accuracy and loss
def plot_training(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))
  
  plt.plot(epochs, acc, 'b-',label='train')
  plt.plot(epochs, val_acc, 'r',label="test")
  plt.legend()
  plt.title('Training and validation accuracy')
  plt.savefig('graph_accuracy.png')
  
  plt.figure()
  plt.plot(epochs, loss, 'b-',label='train')
  plt.plot(epochs, val_loss, 'r-',label='test')
  plt.legend()
  plt.title('Training and validation loss')
  #plt.show()
  plt.savefig('graph_loss.png')
  
plot_training(history)
