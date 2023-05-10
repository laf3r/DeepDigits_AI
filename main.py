import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers

assert 'COLAB_TPU_ADDR' in os.environ, 'Missin TPU?'
if('COLAB_TPU_ADDR') in os.environ:
  TF_MASTER = 'grpc://{}'.format(os.environ['COLAB_TPU_ADDR'])
else:
  TF_MASTER = ''
tpu_address = TF_MASTER

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_address)
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)


strategy = tf.distribute.TPUStrategy(resolver)


def create_model():
  return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
  ])


def get_dataset(batch_size, is_training=True):
    split = 'train' if is_training else 'test'
    dataset, info = tfds.load(name='mnist', split=split, with_info= True, as_supervised=True, try_gcs=True)
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255.0
        return image, label
    
    dataset = dataset.map(scale)
    if is_training:
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset



with strategy.scope():
    model = create_model()
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['sparse_categorical_accuracy'])
    model.summary()

  


batch_size = 512
train_dataset = get_dataset(batch_size, True)
validation_dataset = get_dataset(batch_size, False)
with strategy.scope():
    model = create_model()
    model.compile(optimizer='adam', steps_per_execution=50, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['sparse_categorical_accuracy'])
    epochs = 80
    steps_per_epoch = 60000 // batch_size
    validation_steps = 10000 // batch_size
    history = model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=validation_dataset, validation_steps=validation_steps)
  

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)


plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


final_daset = validation_dataset.take(10)
test_images, test_labels = next(iter(final_daset.take(10)))
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Получение предсказаний нейросети для 10 изображений
predictions = model.predict(test_images)

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6),
                         subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flat):
    # Отображение изображения
    ax.imshow(test_images[i])
    # Отображение меток и предсказаний
    true_label = class_names[test_labels[i]]
    pred_label = class_names[np.argmax(predictions[i])]
    if true_label == pred_label:
        ax.set_title("Это: {}, ИИ: {}".format(true_label, pred_label), color='green')
    else:
        ax.set_title("Это: {}, ИИ: {}".format(true_label, pred_label), color='red')

plt.tight_layout()
plt.show()
