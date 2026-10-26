
import argparse
from pathlib import Path
import idx2numpy
import numpy as np
from sklearn.metrics import classification_report


class SoftmaxRegression:
    def __init__(self, epochs=200, lr=0.5):
        self.epochs = epochs
        self.lr = lr
        self.loss_history = []

    @staticmethod
    def softmax(logits):
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        self.classes_, encoded = np.unique(y, return_inverse=True)
        n, d = X.shape
        self.W = np.zeros((d, len(self.classes_)), dtype=np.float32)
        self.bias = np.zeros(len(self.classes_), dtype=np.float32)
        self.loss_history = []
        for epoch in range(self.epochs):
            logits = X @ self.W + self.bias
            shifted = logits - logits.max(axis=1, keepdims=True)

            loss = np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(n), encoded])
            self.loss_history.append(float(loss))
            error = self.softmax(logits)
            error[np.arange(n), encoded] -= 1
            self.W -= self.lr * (X.T @ error) / n
            self.bias -= self.lr * error.mean(axis=0)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.6f}")
        return self

    def predict_proba(self, X):
        return self.softmax(X @ self.W + self.bias)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--train-limit", type=int, default=None, help="Gioi han mau de chay thu")
    args = parser.parse_args()
    if args.epochs <= 0 or args.lr <= 0 or (args.train_limit is not None and args.train_limit <= 0):
        parser.error("epochs, lr va train-limit phai duong")
    data_dir = Path(__file__).resolve().parent / "data"
    X_train = idx2numpy.convert_from_file(str(data_dir / "train-images.idx3-ubyte"))
    y_train = idx2numpy.convert_from_file(str(data_dir / "train-labels.idx1-ubyte"))
    X_test = idx2numpy.convert_from_file(str(data_dir / "t10k-images.idx3-ubyte"))
    y_test = idx2numpy.convert_from_file(str(data_dir / "t10k-labels.idx1-ubyte"))

    X_train = X_train.reshape(len(X_train), -1).astype(np.float32) / 255.0
    X_test = X_test.reshape(len(X_test), -1).astype(np.float32) / 255.0
    if args.train_limit is not None:
        X_train, y_train = X_train[:args.train_limit], y_train[:args.train_limit]
    model = SoftmaxRegression(args.epochs, args.lr).fit(X_train, y_train)
    predicted_labels = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, predicted_labels, digits=4))

if __name__ == "__main__":
    main()
