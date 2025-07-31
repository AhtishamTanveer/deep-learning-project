import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
import joblib

# Step 1: Load Dataset
dataset_path = 'E:/dataset4/Satellite'  # RSI-CB256 dataset path
batch_size = 32
image_size = (64, 64)

train_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

val_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=image_size,
    batch_size=batch_size
)

# Normalize pixel values
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# Step 2: Preprocess and flatten dataset for DBN
def preprocess_images(dataset):
    images = []
    labels = []
    for image_batch, label_batch in dataset:
        images.append(image_batch.numpy())
        labels.append(label_batch.numpy())
    return np.concatenate(images), np.concatenate(labels)

X_train, y_train = preprocess_images(train_dataset)
X_val, y_val = preprocess_images(val_dataset)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

# Step 3: Build and train the deep belief network (DBN) with RBMs
from sklearn.neural_network import BernoulliRBM

rbm1 = BernoulliRBM(n_components=1024, learning_rate=0.01, n_iter=15, verbose=1)
rbm2 = BernoulliRBM(n_components=512, learning_rate=0.01, n_iter=15, verbose=1)
rbm3 = BernoulliRBM(n_components=256, learning_rate=0.005, n_iter=15, verbose=1)
rbm4 = BernoulliRBM(n_components=128, learning_rate=0.005, n_iter=15, verbose=1)

logistic = LogisticRegression(max_iter=2000, solver='lbfgs')

dbn_model = Pipeline(steps=[
    ('rbm1', rbm1),
    ('rbm2', rbm2),
    ('rbm3', rbm3),
    ('rbm4', rbm4),
    ('logistic', logistic)
])

# Step 4: Train DBN with mini-batches
epochs = 12
num_batches = int(np.ceil(X_train_flat.shape[0] / batch_size))
for epoch in range(epochs):
    print(f"\nDBN Training Epoch {epoch+1}/{epochs}")
    for i in tqdm(range(num_batches), desc="Training RBM Layers"):
        batch = X_train_flat[i * batch_size:(i + 1) * batch_size]
        dbn_model.named_steps['rbm1'].partial_fit(batch)
        batch_transformed = dbn_model.named_steps['rbm1'].transform(batch)
        dbn_model.named_steps['rbm2'].partial_fit(batch_transformed)
        batch_transformed = dbn_model.named_steps['rbm2'].transform(batch_transformed)
        dbn_model.named_steps['rbm3'].partial_fit(batch_transformed)
        batch_transformed = dbn_model.named_steps['rbm3'].transform(batch_transformed)
        dbn_model.named_steps['rbm4'].partial_fit(batch_transformed)
    # Fit logistic regression on features
    x_train_feat = dbn_model.named_steps['rbm4'].transform(X_train_flat)
    dbn_model.named_steps['logistic'].fit(x_train_feat, y_train)
    # Evaluate validation accuracy
    val_feat = dbn_model.named_steps['rbm4'].transform(X_val_flat)
    val_accuracy = dbn_model.named_steps['logistic'].score(val_feat, y_val)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

# Save DBN model
joblib.dump(dbn_model, 'dbnSatellite_model_advanced.pkl')

# Step 5: Fine-tune with CNN
cnn_model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(4)  # 4 classes
])

cnn_model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train CNN
history = cnn_model.fit(
    train_dataset,
    epochs=20,
    validation_data=val_dataset,
    callbacks=[early_stopping]
)

cnn_model.save('cnn_finetuned_advanced.keras')

# Evaluate CNN
test_loss, test_acc = cnn_model.evaluate(val_dataset, verbose=2)
print(f'\nFine-tuned CNN Test accuracy: {test_acc}')

# Plot accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

# Step 6: Random Image Prediction and Visualization
class_names = ['cloudy', 'desert', 'green_area', 'water']
plt.figure(figsize=(7, 7))
for image_batch, label_batch in val_dataset.take(1):
    predictions = cnn_model.predict(image_batch)
    predicted_classes = np.argmax(predictions, axis=1)
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(image_batch[i])
        plt.title(f"Pred: {class_names[predicted_classes[i]]}\nTrue: {class_names[label_batch[i]]}")
        plt.axis('off')
        plt.show()