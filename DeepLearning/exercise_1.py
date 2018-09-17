# Exercise 1
import argparse
import pandas
import keras
import sklearn 
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.datasets import mnist
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras import optimizers, regularizers
from util import print_eval
from util import save_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras import backend as K

def read_args():
    parser = argparse.ArgumentParser(description='Exercise 1')
    parser.add_argument('--num_units', nargs='+', default=[100], type=int,
                        help='Number of hidden units of each hidden layer.')
    parser.add_argument('--dropout', nargs='+', default=[0.5], type=float,
                        help='Dropout ratio for every layer.')
    parser.add_argument('--batch_size', type=str, default='model1',
                        help='Number of instances in each batch.')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment, used in the filename'
                             'where the results are stored.')   
    args = parser.parse_args()
    assert len(args.num_units) == len(args.dropout)
    return args

def load_dataset():
    dataset = load_files('dataset/txt_sentoken', shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=42)
    print('\nTraining samples {}, test_samples {}:'.format(
        len(X_train), len(X_test)))

    # Apply the Tfidf vectorizer to create input matrix
    vectorizer = TfidfVectorizer(binary=True, ngram_range=(1, 2),
    	stop_words='english', max_df=0.7, norm='l2', vocabulary=None)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test, y_train, y_test

def main():    
    args = read_args()
    # Reshaping Train and Test Data
    X_train, X_test, y_train, y_test = load_dataset()
    train_examples = X_train.shape[0]
    test_examples = X_test.shape[0]
    input_size = X_train.shape[1]
    print('Vocabulary Size: '+str(input_size))
    batch_size = 50 # For mini-batch gradient descent
    num_classes = 1 # Binary Classification
    epochs = 50

    # Scaling and Reshaping
    #scaler = StandardScaler(with_mean=False)
    #X_train = scaler.fit_transform(X_train)
    #X_test =  scaler.transform(X_test)
    X_train = X_train.reshape(train_examples, input_size)
    X_test = X_test.reshape(test_examples, input_size)

    # Early Stopping to avoid Overfitting
    early_stop = EarlyStopping(monitor='val_loss', 
    	min_delta=0, patience=15, verbose=1, mode='auto')

    # Build the Keras model 
    # Model 1: 
    model1 = Sequential([
        Dense(16, input_shape=(input_size,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dense(1, activation='relu', kernel_regularizer=regularizers.l2(0.01))])
    
    # Model 2: 
    model2 = Sequential([
        Dense(32, input_shape=(input_size,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dense(1, activation='relu', kernel_regularizer=regularizers.l2(0.01))])
    
    # Model 3: 
    model3 = Sequential([
        Dense(16, input_shape=(input_size,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='relu', kernel_regularizer=regularizers.l2(0.01)) 
    ])

    # Model 4: 
    model4 = Sequential([
        Dense(64, input_shape=(input_size,), activation='relu'),
        Dense(1, activation='relu')])

    # Model 5: 
    model5 = Sequential([
        Dense(16, input_shape=(input_size,), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dense(1, activation='relu', kernel_regularizer=regularizers.l2(0.01))])
    

    model = model5

    # Printing Model Resume
    print(model.summary())
    
    # Compiling the Model
    model.compile(loss='binary_crossentropy',
          optimizer='adam', 
          metrics=['accuracy']) 
    plot_model(model, to_file="Modelos/Fig_Model_{}.png".format(args.experiment_name))
    
    # Fitting the Model
    history = model.fit(X_train, y_train, 
          batch_size=batch_size, epochs=epochs, 
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]);

    # List all data in history
    #print(history.history.keys())
    # Summarize history for accuracy
    fig = plt.figure()
    plt.subplot(121)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # Summarize history for loss
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    fig.savefig("Resultados/Analisis_Overfitting_{}.png".format(args.experiment_name))

    # Evaluation of the Model
    predictions = model.predict(X_test)
    rounded = [round(x[0]) for x in predictions]
    scores = model.evaluate(X_test, y_test)
    print('\n')
    print('Test Loss:', scores[0])
    print('Test Accuracy:', scores[1])

    f = open("Precision/Loss_and_Accuracy_{}.txt".format(args.experiment_name),'w')
    f.write("Training samples {}, test_samples {}.\n".format(train_examples, test_examples))
    f.write('Batch Size: '+str(batch_size)+'\n')
    f.write('Epochs: '+str(epochs)+'\n')
    f.write('Test Loss: '+str(scores[0])+'\n')
    f.write('Test Accuracy: '+str(scores[1])+'\n')
    f.close()

    # Saving the Model.
    model.save("Modelos/Modelo_{}.h5".format(args.experiment_name))

    # Saving the Predictions:
    results = pandas.DataFrame(y_test, columns=['True_Label'])
    results.loc[:, 'Predicted'] = predictions
    results.loc[:, 'Rounded'] =rounded
    results.to_csv("Resultados/Predicitions_{}.csv".format(args.experiment_name),
                        index=False)

if __name__ == '__main__':
    main()
    K.clear_session()