from pandas import read_csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from matplotlib import pyplot

def create_RNN(input_shape):
    hidden_units = 3
    dense_units = 32
    model = Sequential()
    # model.add(LSTM(hidden_units, input_shape=input_shape, activation='tanh'))
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation='tanh'))
    dense_act = 'ReLU'
    # model.add(Dense(units=dense_units, activation=dense_act))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
 
# demo_model = create_RNN(2, 1, (3,1), activation=['linear', 'linear'])
df = read_csv('amoni_pred_base.csv', parse_dates=True, index_col='row_date')
# SPLIT_ON = 176400
SPLIT_ON = 18260
values_t = df.iloc[:SPLIT_ON, 0].values
drift_t = df.iloc[:SPLIT_ON, 1].values
ddrift_t = df.iloc[:SPLIT_ON, 2].values

values_p = df.iloc[SPLIT_ON:, 0].values
drift_p = df.iloc[SPLIT_ON:, 1].values
ddrift_p = df.iloc[SPLIT_ON:, 2].values

def get_X(dat, pred, time_steps, steps):
    number_pred = len(dat) // steps
    slices_dat = []
    slice_pred = []
    for i in range(number_pred):
        min_i = i * steps
        max_i = min_i + time_steps
        if max_i >= len(dat):
            break
        slices_dat.append(dat[min_i:max_i])
        slice_pred.append(pred[max_i - 1])
    X = np.array(slices_dat)
    Y = np.array(slice_pred)
    return X, Y

# TIME_STEPS = 32000
# STEPS = 800
TIME_STEPS = 800
STEPS = 10 # or 50¿ 
trainX, trainY = get_X(values_t, ddrift_t, TIME_STEPS, STEPS)
testX, testY = get_X(values_p, ddrift_p, TIME_STEPS, STEPS)

model = create_RNN(input_shape=(TIME_STEPS,1))
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=50, batch_size=64, verbose=2)

# make predictions
train_predict = model.predict(trainX)
test_predict = model.predict(testX)

_, train_acc = model.evaluate(trainX, trainY, verbose=0)
_, test_acc = model.evaluate(testX, testY, verbose=0)
# Plot the result
def plot_result(trainY, testY, train_predict, test_predict):
    actual = np.append(trainY, testY)
    predictions = np.append(train_predict, test_predict)
    apply = np.vectorize(lambda x: 1 if x > 0.5 else 0)
    binary_pred = apply(predictions)
    rows = len(actual)
    plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(range(rows), actual)
    plt.plot(range(rows), predictions)
    plt.plot(range(rows), binary_pred)
    plt.axvline(x=len(trainY), color='r')
    plt.legend(['Actual', 'Predictions', 'Binary Predictions'])
    plt.xlabel('Observation number after given time steps')
    plt.ylabel('Sunspots scaled')
    plt.title('Actual and Predicted Values. The Red Line Separates The Training And Test Examples')

model.save_weights('save_model')
plot_result(trainY, testY, train_predict, test_predict)
plt.show()

print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()