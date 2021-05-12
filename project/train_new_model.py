import tensorflow as tf
from utilities import *
from modified_unet import *
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os

EXPORT_SUB_DIR="/export/trained_model"
CURRENT_DIR=os.getcwd()
MODEL_EXPORT=CURRENT_DIR+EXPORT_SUB_DIR
model_save = os.path.dirname(MODEL_EXPORT)

dataset, info = tfds.load('oxford_iiit_pet', download=True, with_info=True)

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 2
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

for image, mask in train.take(1):
    sample_image, sample_mask = image, mask

display([sample_image, sample_mask])

tf.keras.utils.plot_model(model, show_shapes=True)

SUB_CHECKPOINT_DIR="training_checkpoint"
CHECKPOINT_DIR=CURRENT_DIR+SUB_CHECKPOINT_DIR
checkpoint=os.path.dirname(CHECKPOINT_DIR)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint,
                                                 save_weights_only=True,
                                                 verbose=1)

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
                create_mask(model.predict(sample_image[tf.newaxis, ...],
                							callbacks=[cp_callback]))])

with tf.device('/CPU:0'):
    
    show_predictions()

    class DisplayCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            show_predictions()
            print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

    EPOCHS = 20
    VAL_SUBSPLITS = 5
    VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS
    
    model_history = model.fit(train_dataset, epochs=EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_steps=VALIDATION_STEPS,
                            validation_data=test_dataset,
                            callbacks=[DisplayCallback()])

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    epochs = range(EPOCHS)

    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    show_predictions(test_dataset, 3)
    tf.saved_model.save(model, model_save)
