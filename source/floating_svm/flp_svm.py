import numpy as np

class FlpSVM(object):

    def __init__(self, lambd=4, lr=0.01, epochs=100) -> None:
        super().__init__()
        self.lambd = lambd
        self.lr = lr
        self.W = None
        self.epochs = epochs

    def fit(self, X, y, verbose=0):
        # Number of features
        m = X.shape[1]
        
        data = np.copy(X)

        # Vector of weights
        self.W = np.random.random(size=(m, 1))
        self.b = np.random.rand()
        y_data = np.copy(y)

        # Losses array
        self.losses = []

        for epoch in range(self.epochs):
            grad_W, grad_b = self.compute_loss_grad(data, y_data)
            self.W = self.W - self.lr * grad_W
            self.b = self.b - self.lr * grad_b 
            
            if verbose == 1:
                print("\t - Epoch:", epoch, " - Cost:", self.loss(data, y_data))

            # To plot the losses
            # self.losses.append(self.loss(data, y_data))

        return self.W                    

    def predict(self, X):
        b_vector = self.b * np.ones((X.shape[0], 1)) 
        return np.sign(np.dot(X, self.W) + b_vector) 

    def loss(self, X, y):
        distances = 1 - np.multiply(y, (np.dot(X, self.W) + self.b))
        distances[distances < 0] = 0

        # Compute of Hinge loss
        hinge_loss = np.sum(distances)

        # Calculate cost
        cost = (self.lambd / 2) * np.dot(self.W.T, self.W) + hinge_loss

        return cost.item()

    def compute_loss_grad(self, X, y):
        distance = 1 - np.multiply(y, (np.dot(X, self.W) + self.b))
        
        dw = np.zeros((len(self.W), 1))
        for index, dist in enumerate(distance):
            if dist <= 0:
                dist_i_w = 0
            else:
                dist_i_w = -y[index][0] * np.expand_dims(X[index], axis=1)

            dw += dist_i_w

        dw += self.lambd * self.W

        db = 0
        for index, dist in enumerate(distance):
            if max(0, dist) == 0:
                dist_i_b = 0
            else:
                dist_i_b = - y[index][0]

            db += dist_i_b

        return dw, db

    def score(self, X, y_true):
        predict = self.predict(X)
        n_correct = np.sum(predict == y_true)
        return n_correct / X.shape[0]

    def load_parameters(self, W, b):
        self.W = W
        self.b = b





