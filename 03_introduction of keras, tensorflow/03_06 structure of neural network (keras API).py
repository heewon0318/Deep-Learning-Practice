import tensorflow as tf
from tensorflow import keras

class SimpleDense(keras.layers.Layer):
    def __int__(self, units, activation = None):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.units),
                                 initializer='random_normal')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros')

    def call(self, inputs):
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y

import numpy as np

model = keras.Sequential([keras.layers.Dense()])
model.compile(optimizer= keras.optimizers.RMSprop(learning_rate= 0.1),
              loss = keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.BinaryAccuracy()])

indices_permutation = np.random.permutation(len(inputs))
shuffled_inputs = inputs[indices_permutation]
shuffled_targets = targets[indices_permutation]

num_validation_samples = int(0.3 * len(inputs))
val_inputs = shuffled_inputs[:num_validation_samples]
val_targets = shuffled_targets[:num_validation_samples]
training_inputs = shuffled_inputs[num_validation_samples : ]
training_targets = shuffled_targets[num_validation_samples : ]
model.fit(training_inputs,
          training_targets,
          epchs = 5,
          batch_size= 16,
          validation_data = (val_inputs, val_targets))

predictions = model.predict(val_inputs, batch_size= 128)
print(predictions[:10])