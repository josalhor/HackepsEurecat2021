from pandas import read_csv
import numpy as np
from sklearn.ensemble import IsolationForest
import math
import matplotlib.pyplot as plt

# demo_model = create_RNN(2, 1, (3,1), activation=['linear', 'linear'])
df = read_csv('amoni_pred.csv', parse_dates=True, index_col='row_date')
SPLIT_ON = 176400
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
        print(i, min_i, max_i, dat[min_i:max_i], pred[max_i - 1])
        slices_dat.append(dat[min_i:max_i])
        slice_pred.append(pred[max_i - 1])
    X = np.array(slices_dat)
    Y = np.array(slice_pred)
    return X, Y


rng = np.random.RandomState(42)
clf = IsolationForest(max_samples=100, random_state=rng)
TIME_STEPS = 32000
STEPS = 800
X_train, Y_train = get_X(values_t, ddrift_t, TIME_STEPS, STEPS)
X_test, Y_test = get_X(values_p, ddrift_p, TIME_STEPS, STEPS)

clf.fit(X_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
# y_pred_outliers = clf.predict(X_outliers)

# Plot the result
def plot_result(trainY, testY, train_predict, test_predict):
    actual = np.append(trainY, testY)
    predictions = np.append(train_predict, test_predict)
    rows = len(actual)
    plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(range(rows), actual)
    plt.plot(range(rows), predictions)
    plt.axvline(x=len(trainY), color='r')
    plt.legend(['Actual', 'Predictions'])
    plt.xlabel('Observation number after given time steps')
    plt.ylabel('Sunspots scaled')
    plt.title('Actual and Predicted Values. The Red Line Separates The Training And Test Examples')

# model.save_weights('save_model')
plot_result(Y_train, Y_test, y_pred_train, y_pred_test)
plt.show()