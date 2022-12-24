import tensorflow as tf
import numpy as np
import os

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(buffer_size=1024)
train_dataset = train_dataset.batch(128)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(128)
test_dataset = test_dataset.cache()
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

print(model.summary())

history = model.fit(train_dataset,
                    validation_data=test_dataset,
                    batch_size=128,
                    epochs=10,
                    )

losses = model.evaluate(test_dataset)
print(losses)

weights = model.get_weights()

new_weights = []
for w in weights:
    if w.ndim != 2:
        new_weights.append(np.expand_dims(w, axis=0))
    else:
        new_weights.append(w)
for w in new_weights:
    print(w.shape)

weight_path = "./data"
def output_weights_and_biases(weights):
    # Output layer1 weight
    w1 = weights[0] * 1e8  # type(w1) = numpy.ndarray
    w1 = w1.astype('int32')
    # To handle negative number:
    # encode if negative: make it absolute and add 8000000(hexadecimal) or 134217728(decimal)
    # decode if number > 134217728(decimal): first minus 134217728(decimal), then make it times (-1)
    w1 = np.where(w1 >= 0, w1, np.abs(w1) + 134217728)
    # output as hex file (%x means hexadecimal)
    w1.tofile(os.path.join(weight_path, 'layer1_weight.hex'), sep=' ', format='%x')
    
    # Output layer1 bias
    b1 = weights[1] * 1e8
    b1 = b1.astype('int32')
    b1 = np.where(b1 >= 0, b1, np.abs(b1) + 134217728)
    b1.tofile(os.path.join(weight_path, 'layer1_bias.hex'), sep=' ', format='%x')
    
    # Output layer2 weight
    w2 = weights[2] * 1e8
    w2 = w2.astype('int32')
    w2 = np.where(w2 >= 0, w2, np.abs(w2) + 134217728)
    w2.tofile(os.path.join(weight_path, 'layer2_weight.hex'), sep=' ', format='%x')
    
    # Output layer2 bias
    b2 = weights[3] * 1e8
    b2 = b2.astype('int32')
    b2 = np.where(b2 >= 0, b2, np.abs(b2) + 134217728)
    b2.tofile(os.path.join(weight_path, 'layer2_bias.hex'), sep=' ', format='%x')

output_weights_and_biases(new_weights)