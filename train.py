import json 
from util import tokenize,stem,bag_of_words
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
# from model import NeuralNet
import pickle

kb_file = 'Data\KB.json'
with open(kb_file, 'r') as f:
    intents=json.load(f)

all_words=[]
tags=[]
xy=[]

for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))


ignore_words=['?','!',',','.']
all_words=[stem(w) for w in all_words if w not in ignore_words]
all_words=sorted(set(all_words))
tags=sorted(set(tags))

X_train=[]
Y_train=[]

for (pattern_sentence,tag) in xy:
    bag =bag_of_words(pattern_sentence,all_words)
    X_train.append(bag)
    label=tags.index(tag)
    Y_train.append(label)

X_train=np.array(X_train)
Y_train=np.array(Y_train)


# Save additional data structures using pickle
# data = {
#     'words': all_words,
#     'classes': tags,
#     'train_x': X_train,  # Convert numpy array to list
#     'train_y': Y_train   # Convert numpy array to list
# }

# with open("training_data.pkl", "wb") as f:
#     pickle.dump(data, f)

# print("Data structures saved to training_data.pkl")



class ChatDataset:
    def __init__(self, X_train, y_train):
        self.X_data = X_train
        self.y_data = y_train
        self.num_samples = len(X_train)

    def __getitem__(self, index):
        # Return the features and label at the specified index
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        # Return the total number of samples in the dataset
        return self.num_samples
    
    
    def to_tf_dataset(self, batch_size=8, shuffle=True):
        # Create a TensorFlow Dataset from numpy arrays
        dataset = tf.data.Dataset.from_tensor_slices((self.X_data, self.y_data))
        
        # Shuffle the dataset if needed
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.num_samples)
        
        # Batch the dataset
        dataset = dataset.batch(batch_size)
        
        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

# Create dataset object
dataset = ChatDataset(X_train, Y_train)

# Create TensorFlow Dataset (equivalent to PyTorch's DataLoader)
train_dataset = dataset.to_tf_dataset(
    batch_size=32,  # Batch size of 32
    shuffle=True    # Shuffle the data
)

hidden_size=64
num_classes=len(tags)
input_size=len(X_train[0])


# Initialize the model using Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=(input_size,)),  # First fully connected layer
    tf.keras.layers.Dense(hidden_size, activation='relu'),  # Second fully connected layer
    tf.keras.layers.Dense(num_classes)  # Output layer with num_classes units
])

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available and will be used.")
    device = '/GPU:0'  # Use the first GPU
else:
    print("No GPU found. Using CPU.")
    device = '/CPU:0'  # Use CPU



# Compile the model with an adjusted learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
with tf.device(device):
    history = model.fit(
        train_dataset,
        epochs=40  # Adjust the number of epochs as needed
    )


batch_size = 100
X_batch = X_train[:batch_size]
Y_batch = Y_train[:batch_size]

# Evaluate the model on the batch
loss, accuracy = model.evaluate(X_batch, Y_batch)

print(f"Batch Loss: {loss}")
print(f"Batch Accuracy: {accuracy}")

# Save the model in the same directory with a .h5 extension
model_save_path = 'my_model.h5'
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Load the model
loaded_model = tf.keras.models.load_model(model_save_path)
print("Model loaded from", model_save_path)

# Print training history
print("Training history:")
print(history.history)

