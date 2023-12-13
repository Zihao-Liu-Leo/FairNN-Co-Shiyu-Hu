import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from pulp import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import argparse
from tqdm import tqdm

def resample(X_train, y_train, c1 = 1, c2 = 1):
    B = X_train['B']
    B_std = (B - np.mean(B)) / np.std(B)
    B_w = np.abs(B_std) + c1
    M = y_train
    M_std = (M - np.mean(M)) / np.std(M)
    M_m = np.sum(B_w * M_std) / np.sum(B_w)
    W = B_w / (np.abs(M_std - M_m) + c2)
    W = W / np.sum(W)
    return W

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def main(cfg):
    data = pd.read_csv('boston.csv')
    features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PIRATIO', 'B', 'LSTAT']
    target = 'MEDV'
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.resample:
        choices = np.random.choice(a = X_train.index, size = len(X_train), replace = True, p = resample(X_train, y_train))
        choices = sorted(list(choices))
        X_train = X_train.loc[choices].reset_index(drop = True)
        y_train = y_train.loc[choices].reset_index(drop = True)
    X_resample = pd.concat([X_train, X_test])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resample)
    X_scaled_df = pd.DataFrame(X_scaled, columns = features)
    X_scaled_df.to_csv('X_scaled.csv', index = False)

    train_len = len(X_train)
    test_len = len(X_test)
    X_train = X_scaled[:train_len, :]
    X_test = X_scaled[train_len:, :]

    mlp = MLPRegressor(hidden_layer_sizes = (100, 50), max_iter = 1000, activation = 'relu', solver = 'adam', random_state = cfg.seed)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    X_test_df = pd.DataFrame(X_test, columns=features)
    y_test_df = pd.DataFrame(y_test, columns=[target])
    y_pred_df = pd.DataFrame(y_pred, columns=['Predicted_' + target])

    results_df = pd.concat([X_test_df, y_test_df, y_pred_df], axis=1)
    results_df.to_csv('test_predictions.csv', index=False)

    X_test_df.to_csv('X_test.csv', index=False)
    y_test_df.to_csv('y_test.csv', index=False)
    y_pred_df.to_csv('y_pred.csv', index=False)

    data = pd.read_csv('X_scaled.csv')
    target = pd.read_csv('y_test.csv')
    X = data.values
    y = target.values.flatten()
    
    input_size = X.shape[1]
    np.random.seed(cfg.seed)
    #w = np.random.randn(input_size)
    #b = np.random.randn()
    w = np.array([0.28967165, -0.00571129, 0.54370172, -1.6018582, -0.30212816, 0.41453689, 
              0.69039547, -1.52664053, 0.19561402, -1.11727328, -1.07267271, -0.70423767, 
              -1.00198178])
    b = -1.5274488644536222
    print('w = {0}\nb = {1}'.format(w, b))
    
    M = 16
    z_segments = np.linspace(-5, 5, M + 1)
    midpoints = (z_segments[:-1] + z_segments[1:]) / 2
    derivative_values = sigmoid(midpoints) * (1 - sigmoid(midpoints))
    lower_slopes = derivative_values
    
    def lower_sigmoid(z):
        z = np.asarray(z)
        y_lower = np.zeros_like(z)
        for i in range(M):
            mask = (z >= z_segments[i]) & (z < z_segments[i+1])
            y_lower[mask] = lower_slopes[i] * (z[mask] - z_segments[i])
        y_lower[z >= z_segments[-1]] = 1
        y_lower[z < z_segments[0]] = 0
        return y_lower
    
    def forward_pass_lower(X, w, b):
        z = np.dot(X, w) + b
        y_pred_lower = 5 + 45*lower_sigmoid(z)
        return y_pred_lower
    
    A = [{'RM': 0.14444444444444446, 'LSTAT': 0.13333333333333336, 'INDUS': 0.12222222222222222,
    'NOX': 0.11111111111111109, 'CRIM': 0.09999999999999999, 'AGE': 0.08888888888888889,
    'DIS': 0.07777777777777778, 'TAX': 0.06666666666666668, 'PIRATIO': 0.055555555555555546,
    'ZN': 0.044444444444444446, 'RAD': 0.033333333333333326, 'CHAS': 0.022222222222222223, 'B': 0.0},
    ]

    X_test = pd.read_csv('X_scaled.csv')
    epsilon = 0.3
    delta_values = []
    dict_weights = A[0]
    
    def lp_norm(x_prime, x_double_prime, p = 2):
        return sum(dict_weights[key]*(x_prime[key] - x_double_prime[key])**p for key in dict_weights)**(1/p)

    def compute_delta(w, b, epsilon, print_data = True):
        y_pred_lower = forward_pass_lower(X, w, b)
        prob = LpProblem("Maximize_delta", LpMaximize)
        dict_weights = A[0]
        variables_x_prime = LpVariable.dicts("x_prime", range(len(X_test)), 0, len(X_test)-1, cat = 'Integer')
        variables_x_double_prime = LpVariable.dicts("x_double_prime", range(len(X_test)), 0, len(X_test)-1, cat = 'Integer')
        delta = LpVariable("delta", lowBound=0)
        prob += -delta
        for i in range(len(X_test)):
                for j in range(i+1, len(X_test)):
                    x_prime_row = X_test.iloc[i]
                    x_double_prime_row = X_test.iloc[j]               
                    dfair_val = lp_norm(x_prime_row, x_double_prime_row)
                    if dfair_val <= epsilon:               
                        delta_val = abs(y_pred_lower[i] - y_pred_lower[j])            
                        prob += delta >= delta_val
        prob.solve(PULP_CBC_CMD(msg=False))
        optimal_delta = value(delta)
        delta_values.append(optimal_delta)
        if print_data : print(f"Epsilon: {epsilon}, Optimal delta: {optimal_delta}")
        return(optimal_delta)
    
    def estimate_gradient(w, b, epsilon, optimal_delta, h=1e-4):
        grad_w = np.zeros_like(w)
        grad_b = 0

        for i in tqdm(range(len(w))):
            w_plus = np.copy(w)
            w_plus[i] += h
            delta_plus = compute_delta(w_plus, b, epsilon, print_data = False)
            w_minus = np.copy(w)
            w_minus[i] -= h
            delta_minus = compute_delta(w_minus, b, epsilon, print_data = False)
            grad_w[i] = (delta_plus - delta_minus) / (2 * h)
            
        b_plus = b + h
        delta_plus = compute_delta(w, b_plus, epsilon, print_data = False)
        b_minus = b - h
        delta_minus = compute_delta(w, b_minus, epsilon, print_data = False)
        grad_b = (delta_plus - delta_minus) / (2 * h)

        return grad_w, grad_b
    
    def gradient_descent(w, b, epochs, lr, epsilon):
        optimal_delta = compute_delta(w, b, epsilon)
        print(f"Epoch 0, delta: {optimal_delta}")
        for epoch in range(epochs):
            grad_w, grad_b = estimate_gradient(w, b, epsilon, optimal_delta, h=1e-4)
            w -= lr * grad_w
            b -= lr * grad_b
            optimal_delta = compute_delta(w, b, epsilon)
            print(f"Epoch {epoch + 1}, delta: {optimal_delta}")
        return w, b
    
    lr = cfg.lr
    epochs = cfg.epochs
    epsilon = 0.3
    
    w_g, b_g = gradient_descent(w, b, epochs, lr, epsilon)
    
    print('w_g = {0},\nb_g = {1}'.format(w_g, b_g))
    return w_g, b_g

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Test')

    parser.add_argument('--resample', type = int, default = 0, help = 'Whether to apply resample')
    parser.add_argument('--seed', type = int, default = 42, help = 'Random seed')
    parser.add_argument('--epochs', type = int, default = 0, help = 'Number of epochs')
    parser.add_argument('--lr', type = float, default = 0.01, help = 'Learning Rate in Sim-grad')

    cfg  = parser.parse_args()
    print(cfg, '\n')

    main(cfg)
