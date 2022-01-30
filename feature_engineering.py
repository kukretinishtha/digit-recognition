from tensorflow.keras.utils import to_categorical

def feature_normalization(dataframe, number_of_pixels):
    dataframe = dataframe.astype('float32')
    dataframe /= number_of_pixels
    return dataframe

def convert_labels_to_one_hot_encoded(labels_dataframe, number_of_labels):
    labels_dataframe = to_categorical(labels_dataframe, number_of_labels)
    return labels_dataframe