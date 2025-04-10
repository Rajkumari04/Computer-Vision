#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from keras import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout


# In[ ]:


import zipfile
zip_ref = zipfile.ZipFile('/content/DATA_NEW.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()


# In[9]:


train_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/DATA_NEW/train',
    labels='inferred',
    label_mode = 'int',
    batch_size=32,
    image_size=(256,256)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/DATA_NEW/test',
    labels='inferred',
    label_mode = 'int',
    batch_size=32,
    image_size=(256,256))


# In[10]:


# Normalize
def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)


# In[11]:


# create CNN model

model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(64,activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(1,activation='sigmoid'))


# In[12]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[13]:


model.summary()


# In[14]:


history = model.fit(train_ds,epochs=10,validation_data=validation_ds)


# In[15]:


import cv2


# In[26]:


test_img = cv2.imread('/content/DATA_NEW/train/dog/dog.920.jpg',cv2.IMREAD_COLOR)


# In[ ]:


import os
output=[]
for i in os.listdir():
  a=cv2.imread(i,cv2.IMREAD_COLOR)
  test_img = cv2.resize(a,(256,256))
  test_input = a.reshape((1,256,256,3))
  output.append(model.predict(test_input))



# In[27]:


plt.imshow(test_img)


# In[18]:


test_img.shape


# In[28]:


test_img = cv2.resize(test_img,(256,256))


# In[22]:


test_img.shape


# In[29]:


plt.imshow(test_img)


# In[30]:


test_input = test_img.reshape((1,256,256,3))


# In[25]:


model


# In[31]:


model.predict(test_input)


# In[ ]:




