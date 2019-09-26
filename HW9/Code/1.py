import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines

def create_dataset():
    #create 1000 data points
    np.random.seed(123)
    uniform_points = np.random.uniform(-5,5,1000)
    uniform_points = uniform_points.reshape(1000)
    uniform_points = np.sort(uniform_points)
    lin_fuc = (3*uniform_points)+2
    noise = np.random.normal(0, 0.1, 1000)
    dataset = lin_fuc + noise
    return (uniform_points, dataset)

def plot_data(data, data_with_noise):
    plt.scatter(data, data_with_noise)
    plt.show()

def find_loss(data, data_with_noise):
    delta=1.0
    MAE = np.absolute(data - data_with_noise)/1000
    huber_mse = 0.5*(data-data_with_noise)**2
    huber_mae = delta * (np.abs(data - data_with_noise) - 0.5 * delta)
    huber = np.where(np.abs(data - data_with_noise) <= delta, huber_mse, huber_mae)
    log_cosh = np.log(np.cosh(data_with_noise - data))
    return MAE, huber, log_cosh

def plot_loss(data, data_with_noise):
    y_pred = data - data_with_noise
    y_pred_logcosh = data_with_noise - data
    loss_mae = np.abs(y_pred)
    loss_huber = np.where(np.abs(y_pred) <= 1.5, 0.5*np.square(y_pred), 1.5 * (np.abs(y_pred) - 0.5 * 1.5))
    loss_logcosh = np.log(np.cosh(y_pred_logcosh))

    #LOSS Visualization
    plt.plot(y_pred, loss_mae, "red", label="MAE")
    plt.plot(y_pred, loss_huber, "green", label="HUBER")
    plt.plot(y_pred_logcosh, loss_logcosh, "blue", label="Log-Cosh")
    plt.legend(loc='upper right')
    plt.grid(True, which="major")
    plt.show()

data, data_with_noise = create_dataset()
plot_data(data, data_with_noise)
MAE, HUBER, LOG_COSH = find_loss(data, data_with_noise)
plot_loss(data, data_with_noise)
