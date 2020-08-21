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
from keras.models import load_model
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras import optimizers 


data_path = "dataset/"
train_dir = data_path + "train/"
test_dir = data_path + "test/"


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


train_datagen = image.ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(256, 256),batch_size=100,class_mode='categorical')

#loading the trained model
model = load_model('dataset/model/final_model_best.h5')

#printing all the layers of the model
#for i, layer in enumerate(model.layers):
#	print(i, layer.name)

test_accuracy = predict_test(model, train_generator)

#printing accuracy on the test dataset
print("Accuracy on test data:",test_accuracy)
