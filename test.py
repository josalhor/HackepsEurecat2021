from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot

import tensorflow as tf
from tensorflow import keras

df = read_csv('amoni_pred_base.csv', parse_dates=True, index_col='row_date')
# SPLIT_ON = 176400
SPLIT_ON = 18260
TIME_STEPS = 1200
STEPS = 1 # or 50¿ 

values_t = df.iloc[:SPLIT_ON, 0].values
drift_t = df.iloc[:SPLIT_ON, 1].values
ddrift_t = df.iloc[:SPLIT_ON, 2].values

values_p = df.iloc[SPLIT_ON - TIME_STEPS:, 0].values
drift_p = df.iloc[SPLIT_ON - TIME_STEPS:, 1].values
ddrift_p = df.iloc[SPLIT_ON - TIME_STEPS:, 2].values

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

def plot_result(trainY, testY, train_predict, test_predict):
    actual = np.append(trainY, testY)
    predictions = np.append(train_predict, test_predict)
    apply = np.vectorize(lambda x: 1 if x > 0.5 else 0)
    binary_pred = apply(predictions)
    rows = len(actual)
    plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(range(rows), actual)
    # plt.plot(range(rows), predictions)
    plt.plot(range(rows), binary_pred)
    plt.axvline(x=len(trainY), color='r')
    plt.legend([
        'Actual',
        # 'Predictions',
        'Binary Predictions'
    ])
    plt.xlabel('Observation number after given time steps')
    plt.ylabel('Sunspots scaled')
    plt.title('Actual and Predicted Values. The Red Line Separates The Training And Test Examples')
    from sklearn.metrics import accuracy_score
    train_acc = accuracy_score(trainY, apply(train_predict))
    test_acc = accuracy_score(testY, apply(test_predict))
    print(f'Train: Acc - {train_acc} | Test: Acc - {test_acc}')
    plt.show()
    return binary_pred



for ON in ['dangerous_drift', 'drift']:
    if ON == 'dangerous_drift':
        model = tf.keras.models.load_model('save_dd.h5')
        t1 = ddrift_t
        t2 = ddrift_p
    elif ON == 'drift':
        model = tf.keras.models.load_model('save_d4.h5')
        t1 = drift_t
        t2 = drift_p

    trainX, trainY = get_X(values_t, t1, TIME_STEPS, STEPS)
    testX, testY = get_X(values_p, t2, TIME_STEPS, STEPS)
    print(len(trainY) + len(testY), len(t1), len(t2), len(df))

    # make predictions
    train_predict = model.predict(trainX)
    test_predict = model.predict(testX)


    binary_pred = plot_result(trainY, testY, train_predict, test_predict)
    padded_pred = np.append(np.zeros((TIME_STEPS,)), binary_pred)
    df[f'pred_{ON}'] = padded_pred

df.to_csv('output.csv')