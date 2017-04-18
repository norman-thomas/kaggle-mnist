

```python
import numpy as np
import pandas as pd

import math

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Lambda, Flatten, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib
import seaborn as sns

%matplotlib inline
sns.set(color_codes=True, palette='muted')
```

## Glimpse at the MNIST Data


```python
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

df_train = pd.read_csv(TRAIN_FILE)
df_test = pd.read_csv(TEST_FILE)
```

### Getting an Overview


```python
print('Training data:', df_train.info())
print('\n --- \n')
print('Test data:', df_test.info())
```


```python
df_train.describe()
```


```python
df_test.describe()
```


```python
df_train.head()
```


```python
df_train.tail()
```

### Splitting into Features and Labels


```python
y_train = to_categorical(df_train['label'].values)
X_train = df_train.drop('label', axis=1).values
X_test = df_test.values

n_features = len(X_train[0])
n_pixels = int(math.sqrt(n_features))
n_classes = y_train.shape[1]

print('We\'ve got {} feature rows and {} labels'.format(len(X_train), len(y_train)))
print('Each row has {} features'.format(len(X_train[0])))
print('and we have {} classes'.format(n_classes))
assert(len(y_train) == len(X_train))
assert(len(X_train[0] == len(X_test[0])))
assert(n_features == n_pixels**2)
print('Input images have {0} x {0} px shape'.format(n_pixels))
print('So far, so good')
```


```python
X_train = X_train.reshape(X_train.shape[0], n_pixels, n_pixels, 1)
X_test = X_test.reshape(X_test.shape[0], n_pixels, n_pixels, 1)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15)

print('X_train.shape:', X_train.shape)
print('y_train.shape:', y_train.shape)
print('X_valid.shape:', X_valid.shape)
print('y_valid.shape:', y_valid.shape)
print('X_test.shape:', X_test.shape)
```


```python
sns.distplot(df_train['label'].values, kde=False, vertical=False, bins=10)
```


```python
def create_my_model(shape=(28, 28, 1)):
    
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=shape))
    
    model.add(Conv2D(16, (5, 5), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3), activation='elu'))
    
    model.add(Flatten())
    model.add(Dropout(0.6))
    
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(32, activation='elu'))
    model.add(Dense(n_classes, activation='softmax'))
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
    
model = create_my_model(shape=(n_pixels, n_pixels, 1))
model.summary()
```


```python
imgen_train = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1, # ???
    zoom_range=0.1, # 0.2 ?
    horizontal_flip=False,
    vertical_flip=False
)

imgen_valid = ImageDataGenerator()

imgen_train.fit(X_train)
imgen_valid.fit(X_valid)
```


```python
EPOCHS = 16
BATCH = 256

#model.fit(X_train, y_train, batch_size=BATCH, epochs=EPOCHS, validation_split=0.2)
model.fit_generator(
    imgen_train.flow(X_train, y_train, batch_size=BATCH),
    steps_per_epoch=4*X_train.shape[0]//BATCH,
    epochs=EPOCHS,
    validation_data=imgen_valid.flow(X_valid, y_valid),
    validation_steps=BATCH
)
```


```python
model.save('model.h5')
```


```python
y_test = model.predict_classes(X_test)
print(y_test.shape)

result = list(enumerate(y_test))
result = pd.DataFrame.from_dict({'ImageId': range(1, len(y_test)+1), 'Label': y_test})
```


```python
result.to_csv('submission.csv', index=False)
```


```python

```
