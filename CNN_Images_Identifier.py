#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## import tensorflow
import tensorflow as tf
from tensorflow.keras import datasets,layers,models
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


(train_images,train_labels),(test_images,test_labels)=datasets.cifar10.load_data()


# In[ ]:


## normalize pixel values to be between 0 and 1
train_images,test_images = train_images/255.0 , test_images/255.0


# In[ ]:


class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
plt.figure(figsize=(8,8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    #THe CIFAR labels happen to be array
    #which is why we need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show


# In[ ]:


model=models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10)
     ])


# In[ ]:


##model.add(layers.Flatten())


##model.add(layers.Dense(64,activation='relu'))

##model.add(layers.Dense(10))


# In[ ]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
## An epoch mea trainig the neurak network with all the training data for one cycle.Here I use 10 epochs
history=model.fit(train_images,train_labels,epochs=10,validation_data=(test_images,test_labels),verbose=0)


# In[ ]:


model.summary()


# In[ ]:


def predict_single_image(model,image_index):
  ##extract the test image and its true label
  img = test_images[image_index]
  true_label=test_labels[image_index][0]


  #Expand dimensions to match model input requirements(batch size)

  img_expanded = np.expand_dims(img,axis=0)

  #get predictions

  predictions = model.predict(img_expanded,verbose=0)
  predicted_label=np.argmax(predictions[0])
  confidence=tf.nn.softmax(predictions[0])

  ## display the image and prediction result

  plt.imshow(img)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.title(f"Predicted: {class_names[predicted_label]} (Confidence:{confidence[predicted_label]:.2f})\n" f"True:{class_names[true_label]}")
  plt.show()
  ## print detailed confidence scoresprint(f"Predicted Label:{class_names[predicted_label]}")
  print(f"Predicted:{class_names[predicted_label]}")
  print(f"True Label:{class_names[true_label]}")
  print("Confidence Scores:")
  for i,score in enumerate(predictions[0]):
    print(f"{class_names[i]}:{score:.4f}")

#7 use the function tot test predictions on a single image
image_index=15 # change this index to test other images
predict_single_image(model,image_index)


# In[ ]:




