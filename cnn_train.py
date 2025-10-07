import os
import sys
import pathlib

import tensorflow as tf
import keras 
from keras import layers, models


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_wav_mono(filename : str):
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)  # type: ignore
    wav = tf.squeeze(wav, axis=1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    return wav

def preprocess(file_path : str, label : str):
    wav = load_wav_mono(file_path)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)  # type: ignore
    wav = tf.concat([zero_padding, wav], 0)
    spectogram = tf.signal.stft(wav, frame_length=160, frame_step = 16)
    spectogram = tf.abs(spectogram)
    print(spectogram.shape)
    spectogram = tf.expand_dims(spectogram, axis=2)
    print(spectogram.shape)
    return spectogram, label

# ensures reproducibliity as neurons are set at a known seed
seed = 7
tf.random.set_seed(seed)
np.random.seed(seed)

main_dir = pathlib.Path(__file__).parent
DATASET_PATH = 'DroneAudioDataset-master'
Binary_Drone_Audio_Folder = 'Binary_Drone_Audio' 
yes_drone_path = 'yes_drone/*.wav'
no_drone_path = 'unknown/*.wav'

POS = os.path.join(main_dir, DATASET_PATH, Binary_Drone_Audio_Folder, yes_drone_path)
NEG = os.path.join(main_dir, DATASET_PATH, Binary_Drone_Audio_Folder, no_drone_path)
pos = tf.data.Dataset.list_files(POS)
neg = tf.data.Dataset.list_files(NEG)

positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))

data = positives.concatenate(negatives)
print("No. Wav: ", len(data))
data = data.map(preprocess)
data = data.cache()
data = data.shuffle(11704)
data = data.batch(16)
data = data.prefetch(4)
print("Data: ", len(data))

train = data.take(32)
test = data.skip(32).take(16)

samples, labels = train.as_numpy_iterator().next()  # type: ignore
print(samples.shape)

num_samples = set(int(i.shape[0]) for i in keras.tree.flatten(samples))

model = models.Sequential(
    [
        layers.Input(shape=(1491, 257, 1)),
        layers.Resizing(1024, 256),

        layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu'),
        layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid'),
    ]
)

model.compile('adam', loss='BinaryCrossentropy', run_eagerly=True, metrics=[keras.metrics.Recall(), keras.metrics.Precision()])
model.summary()

hist = model.fit(train, epochs=5, validation_data=test, verbose='auto')

# Training Stats
plt.title('Loss')
plt.plot(hist.history['loss'], 'r')
plt.plot(hist.history['val_loss'], 'b')
plt.show()

plt.title('Precision')
plt.plot(hist.history['precision'], 'r')
plt.plot(hist.history['val_precision'], 'b')
plt.show()

plt.title('Recall')
plt.plot(hist.history['recall'], 'r')
plt.plot(hist.history['val_recall'], 'b')
plt.show()

print("Test Prediction...")
test = model.predict(samples)
print("Test Prediction result: ", test)
