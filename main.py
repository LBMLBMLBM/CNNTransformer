import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import datetime
import os
from tensorflow.keras.callbacks import ModelCheckpoint

# 定义CNN模型
def build_cnn_model(input_shape, num_classes, filters):
    model = models.Sequential([
        layers.Conv2D(filters, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(filters * 2, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(filters * 4, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(filters * 4, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


# 定义Transformer模型
def build_transformer_model(input_shape, num_classes, num_layers, d_model, num_heads, mlp_dim):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(d_model, kernel_size=(3, 3), strides=(1, 1), padding="same")(inputs)  # Initial embedding
    for _ in range(num_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x1, x1)
        x2 = layers.Add()([attention_output, x])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(mlp_dim, activation='relu')(x3)
        x = layers.Dense(d_model, activation='relu')(x3)
        x = layers.Add()([x, x2])
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


# 数据加载及初始化
def preprocess_data():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 100)
    y_test = to_categorical(y_test, 100)
    return x_train, y_train, x_test, y_test


# CutMix数据增强
def cutmix(image, label, batch_size, alpha=1.0):
    indices = tf.range(start=0, limit=batch_size, dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    shuffled_images = tf.gather(image, shuffled_indices)
    shuffled_labels = tf.gather(label, shuffled_indices)
    image_height, image_width, channels = image.shape[1:]
    cut_ratio = np.random.beta(alpha, alpha)
    cut_height = tf.cast(cut_ratio * tf.cast(image_height, tf.float32), tf.int32)
    cut_width = tf.cast(cut_ratio * tf.cast(image_width, tf.float32), tf.int32)
    y = tf.random.uniform(shape=[], minval=0, maxval=image_height, dtype=tf.int32)
    x = tf.random.uniform(shape=[], minval=0, maxval=image_width, dtype=tf.int32)
    ymin = tf.clip_by_value(y - cut_height // 2, 0, image_height)
    xmin = tf.clip_by_value(x - cut_width // 2, 0, image_width)
    ymax = tf.clip_by_value(y + cut_height // 2, 0, image_height)
    xmax = tf.clip_by_value(x + cut_width // 2, 0, image_width)
    mid_left = image[:, ymin:ymax, 0:xmin, :]
    mid_mid = shuffled_images[:, ymin:ymax, xmin:xmax, :]
    mid_right = image[:, ymin:ymax, xmax:image_width, :]
    middle_concat = tf.concat([mid_left, mid_mid, mid_right], axis=2)
    top = image[:, 0:ymin, :, :]
    bottom = image[:, ymax:image_height, :, :]
    masked_image = tf.concat([top, middle_concat, bottom], axis=1)
    area_ratio = tf.cast((ymax - ymin) * (xmax - xmin) / (image_width * image_height), tf.float32)
    label = (1 - area_ratio) * label + area_ratio * shuffled_labels
    return masked_image, label


# 模型训练函数
def compile_and_train(model, x_train, y_train, x_test, y_test, batch_size, epochs, learning_rate, model_name,
                      param_config):
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=['accuracy'])
    # 用TensorBoard记录模型训练过程中的性能指标
    log_dir = os.path.join("logs", model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True,
                                                 update_freq='batch')

    # 添加ModelCheckpoint回调函数来保存最佳参数
    checkpoint_path = f"best_model_{model_name}_{param_config}.h5"
    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max')

    def train_generator():
        while True:
            indices = np.random.permutation(len(x_train))
            num_batches = len(x_train) // batch_size
            for batch_idx in range(num_batches):
                batch_indices = indices[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                batch_images, batch_labels = x_train[batch_indices], y_train[batch_indices]
                yield cutmix(batch_images, batch_labels, batch_size)

    history = model.fit(train_generator(), steps_per_epoch=len(x_train) // batch_size, epochs=epochs,
                        validation_data=(x_test, y_test), callbacks=[tensorboard_callback, checkpoint_callback])
    return history

os.makedirs('./plots', exist_ok=True)
# 绘图
def plot_history(history, model_name, param_config):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./plots/{model_name}_{param_config}_history.png')
    plt.show()


# 用于评估模型
def evaluate(model, x_test, y_test, classes, model_name):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'{model_name} Test Accuracy: {test_acc}, Test Loss: {test_loss}')


# 超参数
params = [
    {'filters': 32, 'num_layers': 4, 'd_model': 64, 'num_heads': 4, 'mlp_dim': 128, 'learning_rate': 0.01,
     'batch_size': 16, 'epochs': 12},
    {'filters': 32, 'num_layers': 4, 'd_model': 64, 'num_heads': 4, 'mlp_dim': 128, 'learning_rate': 0.001,
     'batch_size': 16, 'epochs': 15},
    {'filters': 32, 'num_layers': 4, 'd_model': 64, 'num_heads': 4, 'mlp_dim': 256, 'learning_rate': 0.0005,
     'batch_size': 16, 'epochs': 50},
    {'filters': 64, 'num_layers': 4, 'd_model': 128, 'num_heads': 4, 'mlp_dim': 128, 'learning_rate': 0.001,
     'batch_size': 16, 'epochs': 50}
]

# 主程序
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = preprocess_data()
    class_names = [str(i) for i in range(100)]
    for param in params:
        param_str = f"filters_{param['filters']}_layers_{param['num_layers']}_dmodel_{param['d_model']}_heads_{param['num_heads']}_mlpdim_{param['mlp_dim']}_batch_{param['batch_size']}_lr_{param['learning_rate']}_epochs_{param['epochs']}"

        cnn_model = build_cnn_model((32, 32, 3), 100, param['filters'])

        print("Training CNN Model")
        cnn_history = compile_and_train(cnn_model, x_train, y_train, x_test, y_test, param['batch_size'],
                                        param['epochs'], param['learning_rate'], 'CNN', param_str)
        plot_history(cnn_history, "CNN", param_str)
        evaluate(cnn_model, x_test, y_test, class_names, f"CNN_{param_str}")

        transformer_model = build_transformer_model((32, 32, 3), 100, param['num_layers'], param['d_model'],
                                                    param['num_heads'], param['mlp_dim'])
        print("Training Transformer Model")
        transformer_history = compile_and_train(transformer_model, x_train, y_train, x_test, y_test,
                                                param['batch_size'], param['epochs'], param['learning_rate'],
                                                "Transformer", param_str)
        plot_history(transformer_history, "Transformer", param_str)
        evaluate(transformer_model, x_test, y_test, class_names, f"Transformer_{param_str}")

