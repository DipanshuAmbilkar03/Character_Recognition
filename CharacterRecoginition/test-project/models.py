import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.keras.datasets import mnist


# Load dataset
def load_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print(f"Input shape: {x_train.shape[1:]}")
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Number of classes: {np.max(y_train) + 1}")

    return (x_train, y_train), (x_test, y_test)


def create_simple_cnn(input_shape=(28, 28, 1), num_classes=10):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            # Feature extraction
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            # Classification head
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# Main
if __name__ == "__main__":
    # Load data
    (x_train, y_train), (x_test, y_test) = load_dataset()

    # Create model
    model = create_simple_cnn(num_classes=int(np.max(y_train)) + 1)
    model.summary()

    # Train
    history = model.fit(
        x_train, y_train, batch_size=128, epochs=10, validation_split=0.1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Save model
    model.save("char_recognition.keras")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss Curves")
    plt.legend()

    plt.tight_layout()
    plt.show()
