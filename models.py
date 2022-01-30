import pandas as pd
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers

def model_architecture(n_input, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, num_digits):
    input_shape = Input(shape=(n_input,))
    x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(input_shape)
    x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)
    x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)
    x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)
    output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)
    model = Model(input_shape, output)
    return model

def train_model(model, X_train, y_train, X_validation, y_validation,
                loss, optimizer, metrics, batch_size, training_epochs):
    model.compile(loss = 'categorical_crossentropy',optimizer = 'sgd',metrics = 'accuracy')
    history = model.fit(X_train, y_train,
                     batch_size = batch_size,
                     epochs = training_epochs,
                     verbose = 2,
                     validation_data=(X_validation, y_validation))
    return model

def predict_on_test_dataset(model, X_test):
    test_pred = pd.DataFrame(model.predict(X_test, batch_size=200))
    test_pred = pd.DataFrame(test_pred.idxmax(axis = 1))
    test_pred.index.name = 'ImageId'
    test_pred = test_pred.rename(columns = {0: 'Label'}).reset_index()
    test_pred['ImageId'] = test_pred['ImageId'] + 1
    return test_pred

def save_model(model):
    model.save("digit_classification_model.h5")

def save_data_to_csv(dataset, filename='predict.csv'):
    dataset.to_csv(filename, index = False)