import numpy as np 

class LogisticRegression:
    def __init__(self, epoch=100, lr=0.01):
        self.epoch = epoch
        self.lr = lr
        self.W = None
        self.bias = 0.0
        self.loss_history = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)  
        return 1 / (1 + np.exp(-z))


    def loss_fn(self, y_true, y_pred):
        epsilon = 1e-12
        probabilities = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(probabilities) + (1 - y_true) * np.log(1 - probabilities))
        return loss

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        n_samples, n_features = X.shape

        if len(y) != n_samples:
            raise ValueError("X va y phai co cung so luong mau.")

        self.W = np.zeros((n_features, 1), dtype=np.float64)
        self.bias = 0.0

        self.loss_history = []

        for epoch in range(self.epoch):

            z = X @ self.W + self.bias
            prohibilities = self.sigmoid(z)

            loss = self.loss_fn(y, prohibilities)
            self.loss_history.append(loss)

            error = prohibilities - y

            dW = (X.T @ error) / n_samples
            db = np.mean(error)


            self.W -= self.lr * dW
            self.bias -= self.lr * db

            if(epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epoch}, Loss: {loss:.6f}")
        return self


    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        z = X @ self.W + self.bias
        probabilities = self.sigmoid(z).reshape(-1)
        return np.column_stack((1 - probabilities, probabilities))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def evaluate(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, 1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1, 1)

        accuracy = np.mean(y_true == y_pred)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score)}
    

    
        
        

    