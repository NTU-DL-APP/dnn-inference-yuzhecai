import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 讀資料
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 正規化
train_images = train_images / 255.0
test_images = test_images / 255.0

# 模型定義
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 編譯
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callback：提早停止 + 儲存最好的模型
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('./model/fashion_mnist.h5', save_best_only=True)
]

# 訓練
model.fit(train_images, train_labels,
          epochs=50,
          batch_size=128,
          validation_split=0.1,
          callbacks=callbacks)

# 儲存 .h5
model.save('./model/fashion_mnist.h5')