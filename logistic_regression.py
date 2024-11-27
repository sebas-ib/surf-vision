import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, w, b, lambda_=0):
    m = len(y)
    z_wb = np.dot(X, w) + b
    f_wb = sigmoid(z_wb)
    epsilon = 1e-5

    f_wb = np.clip(f_wb, epsilon, 1 - epsilon)
    loss = - ( y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb) )
    total_cost = np.mean(loss)

    reg_cost = (lambda_ / (2 * m)) * np.sum(w ** 2)
    total_cost += reg_cost

    return total_cost

def compute_gradient(X, y, w, b, lambda_=0):
    m = len(y)
    z_wb = np.dot(X, w) + b
    f_wb = sigmoid(z_wb)

    diff = f_wb - y
    dj_dw = (np.dot(X.T, diff) / m) + (lambda_ / m) * w
    dj_db = np.sum(diff) / m

    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    m = len(X)
    J_history = []
    w_history = []

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        if i < 100000:
            cost = cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        if i % (num_iters // 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

    return w_in, b_in, J_history, w_history

class LogisticRegression:
    def __init__(self, alpha=0.01, num_iters=1000, lambda_=0):
        self.alpha = alpha
        self.num_iters = num_iters
        self.lambda_ = lambda_
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape

        self.w = np.random.randn(n) * 0.01
        self.b = 0
        self.w, self.b, _, _ = gradient_descent(
            X, y, self.w, self.b, compute_cost, compute_gradient, self.alpha, self.num_iters, self.lambda_)

    def predict(self, X):
        X = np.array(X)
        z_wb = np.dot(X, self.w) + self.b
        y_pred = (sigmoid(z_wb) >= 0.5).astype(int)
        return y_pred

    def predict_proba(self, X):
        X = np.array(X)
        z_wb = np.dot(X, self.w) + self.b
        y_proba = sigmoid(z_wb)
        return y_proba