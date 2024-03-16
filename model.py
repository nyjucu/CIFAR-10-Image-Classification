from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, ReLU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

X_train, X_test = X_train / 255.0, X_test / 255.0

y_train, y_test = to_categorical(y_train), to_categorical(y_test)

print(y_train.shape)

if __name__ == "__main__":
    datagen = ImageDataGenerator(
        rotation_range=10,
        horizontal_flip=True
    )

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 3)))
    model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    model.add(ReLU())
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    model.add(ReLU())
    model.add(MaxPool2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    model.add(ReLU())
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    model.add(ReLU())
    model.add(MaxPool2D((2, 2)))

    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    model.add(ReLU())
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    model.add(ReLU())
    model.add(MaxPool2D((2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    model.add(ReLU())

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    model.add(ReLU())

    model.add(Dense(10, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(datagen.flow(X_train, y_train, batch_size=128), epochs=32, steps_per_epoch=len(X_train)//128)

    _, accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy of the model: {accuracy}")

    model.save("classification.keras")
