#------------------- 1066600 --------------------#
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# Load MNIST Data
def Load_data():
    # Load Train Data   
    Train_data = np.loadtxt('sign_mnist_train.csv', dtype = 'float', delimiter = ',');
    Train_Label = Train_data[:,0]
    # Load Test Data
    Test_data = np.loadtxt('sign_mnist_test.csv', dtype = 'float', delimiter = ',');
    Test_Label = Test_data[:,0]
    #---#
    return Train_data[:,1:]/255, Test_data[:,1:]/255, Train_Label[:], Test_Label[:]

if __name__ == "__main__":
    # Load
    Train_data, Test_data, Train_Label, Test_Label = Load_data()

    # Neural network
    model = Sequential()
    model.add(Dense(300, input_dim=784, activation='relu'))
    model.add(Dense(25, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='accuracy', patience=3)
    # Train
    history = model.fit(Train_data, Train_Label, epochs=100, batch_size=1000, verbose = 1,
                        shuffle=True, callbacks =[early_stopping])

    # Test
    Per = model.evaluate(Test_data,Test_Label,batch_size=1000)
    print(Per)
    
    # Extract weights and biases
    A1 = model.layers[0].get_weights()[0]
    B1  = model.layers[0].get_weights()[1]
    A2 = model.layers[1].get_weights()[0]
    B2  = model.layers[1].get_weights()[1]

    # Save to txt
    np.savetxt('A1.csv', A1, delimiter = ',')
    np.savetxt('A2.csv', A2, delimiter = ',')
    np.savetxt('B1.csv', B1, delimiter = ',')
    np.savetxt('B2.csv', B2, delimiter = ',')
