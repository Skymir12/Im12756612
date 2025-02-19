import pathlib  # Работа с файловыми путями
import matplotlib.pyplot as plt  # Библиотека для визуализации
from matplotlib.figure import Figure  
import numpy as np  # Работа с массивами
import PIL  # Работа с изображениями
import tensorflow as tf  # Основная библиотека для машинного обучения

from tensorflow import keras
from tensorflow.keras import layers  # Импорт слоев для нейросети
from tensorflow.keras.models import Sequential  # Последовательная модель

# Указываем путь к папке с изображениями для обучения
dataset_dir = pathlib.Path("Training")

# Определяем параметры загрузки изображений
batch_size = 32  # Размер пакета (количество изображений за 1 итерацию)
img_width = 180  # Ширина изображения
img_height = 180  # Высота изображения

# Загружаем тренировочный датасет из папки, используя функцию Keras
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,  # 20% данных выделяем на валидацию
    subset="training",  # Загружаем только тренировочные данные
    seed=123,  # Фиксируем случайность для воспроизводимости
    image_size=(img_height, img_width),  # Изменяем размер изображений
    batch_size=batch_size  # Задаем размер пакета
)

# Загружаем валидационный датасет из той же папки
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Получаем список классов (категорий изображений)
class_names = train_ds.class_names
print(f'Class names: {class_names}')  # Выводим классы

# Оптимизируем загрузку данных для увеличения производительности
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Количество классов в наборе данных
num_classes = len(class_names)

# Создаем модель нейросети
model = Sequential([
    # Нормализация входных данных (приводим к диапазону [0,1])
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    # Аугментация данных (изменения для увеличения разнообразия)
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),  # Отражение по горизонтали
    layers.experimental.preprocessing.RandomRotation(0.1),  # Случайный поворот на 10%
    layers.experimental.preprocessing.RandomZoom(0.1),  # Масштабирование
    layers.experimental.preprocessing.RandomContrast(0.2),  # Контрастность

    # Сверточный слой 1 (16 фильтров 3x3) + max pooling
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    # Сверточный слой 2 (32 фильтра 3x3) + max pooling
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    # Сверточный слой 3 (64 фильтра 3x3) + max pooling
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    # Dropout (отключение случайных нейронов для предотвращения переобучения)
    layers.Dropout(0.2),

    # Выравнивание (Flatten) и полносвязные слои
    layers.Flatten(),
    layers.Dense(128, activation='relu'),  # Полносвязный слой (128 нейронов)
    layers.Dense(num_classes)  # Выходной слой (число классов)
])

# Компиляция модели
model.compile(
    optimizer='adam',  # Оптимизатор Adam (адаптивный градиентный спуск)
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Функция потерь
    metrics=['accuracy']  # Отслеживаем точность
)

# Вывод структуры модели
model.summary()

# Количество эпох обучения
epochs = 10

# Обучение модели
history = model.fit(
    train_ds,
    validation_data=val_ds,  # Проверка на валидационных данных
    epochs=epochs  # Количество эпох
)

# Извлекаем историю обучения (точность и потери)
acc = history.history['accuracy']  # Точность на обучающей выборке
val_acc = history.history['val_accuracy']  # Точность на валидационной выборке

loss = history.history['loss']  # Ошибки на обучении
val_loss = history.history['val_loss']  # Ошибки на валидации

epochs_range = range(epochs)  # Создаем диапазон значений (эпох)

# Визуализируем результаты обучения
Figure(figsize=(8, 8))  # Размер графиков

plt.subplot(1, 2, 1)  # Первый график
plt.plot(epochs_range, acc, label='Training Accuracy')  # Точность на обучении
plt.plot(epochs_range, val_acc, label='Validation Accuracy')  # Точность на валидации
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)  # Второй график
plt.plot(epochs_range, loss, label='Training Loss')  # Ошибка на обучении
plt.plot(epochs_range, val_loss, label='Validation Loss')  # Ошибка на валидации
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()  # Показываем графики

# Сохраняем веса обученной модели
model.save_weights("FlowerModel")
print("Модель сохранена")