from preprocessing import read_dataset, splitting_dataset_into_testing_and_validation, reshape_matrix
from feature_engineering import feature_normalization, convert_labels_to_one_hot_encoded
from models import model_architecture, train_model, predict_on_test_dataset, save_model, save_data_to_csv
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
train_dataset_filepath = 'dataset/train.csv'
test_dataset_filepath = 'dataset/test.csv' 
number_of_features = 785
number_of_pixels = 255
number_of_labels = 10
label = 0

# Input Parameters
n_input = 784 
n_hidden_1 = 300
n_hidden_2 = 100
n_hidden_3 = 100
n_hidden_4 = 200
num_digits = 10
hidden_layer_activation_function = 'relu'
output_layer_activation_function = 'softmax'
# Insert Hyperparameters
learning_rate = 0.1
training_epochs = 20
batch_size = 100
sgd = optimizers.SGD(lr=learning_rate)
#Model parameters
loss = 'categorical_crossentropy'
optimizer = 'sgd'
metrics = 'accuracy'

# Read train and test dataset
train_dataset = read_dataset(train_dataset_filepath)
test_dataset = read_dataset(test_dataset_filepath)
print(f'--------------> train_dataset <---------------')
print('Number of rows: {}'.format(test_dataset.shape[0]))
print('Number of columns: {}'.format(test_dataset.shape[1]))
# Split train dataset to train and validation dataset
X_train, X_validation, y_train, y_validation = splitting_dataset_into_testing_and_validation(train_dataset, label, number_of_features)
# Reshaping dataset
X_train = reshape_matrix(X_train, X_train.shape[0], X_train.shape[1])
X_validation = reshape_matrix(X_validation, X_validation.shape[0], X_validation.shape[1])
# Feature normalization
X_train = feature_normalization(X_train, number_of_pixels)
X_validation = feature_normalization(X_validation, number_of_pixels)
# One hot encoding for target dataset
y_train = convert_labels_to_one_hot_encoded(y_train, number_of_labels)
y_validation = convert_labels_to_one_hot_encoded(y_validation, number_of_labels)
print('--------------> X_train <---------------')
print(X_train)
print('------------------> X_validation <---------------')
print(y_train)
# Get model architeture
model = model_architecture(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, num_digits)
trained_model = train_model(
    model, X_train, y_train, X_validation, y_validation,
    loss, optimizer, metrics, batch_size, training_epochs
)
# Save the trained model
save_model(trained_model)
# Preprocessing X_test
X_test = reshape_matrix(test_dataset, test_dataset.shape[0], test_dataset.shape[1])
X_test = feature_normalization(X_test, number_of_pixels)
test_dataset = predict_on_test_dataset(model, X_test)
save_data_to_csv(test_dataset, 'mnist_test_prediction.csv')