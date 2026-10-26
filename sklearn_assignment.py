import argparse
import inspect
from pathlib import Path

import idx2numpy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



def evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, digits=4))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--train-limit", type=int, default=None, help="Gioi han mau de chay thu")
    args = parser.parse_args()
    if args.max_iter <= 0 or (args.train_limit is not None and args.train_limit <= 0):
        parser.error("max-iter va train-limit phai duong")
    data_dir = Path(__file__).resolve().parent / "data"
    X_train = idx2numpy.convert_from_file(str(data_dir / "train-images.idx3-ubyte"))
    y_train = idx2numpy.convert_from_file(str(data_dir / "train-labels.idx1-ubyte"))
    X_test = idx2numpy.convert_from_file(str(data_dir / "t10k-images.idx3-ubyte"))
    y_test = idx2numpy.convert_from_file(str(data_dir / "t10k-labels.idx1-ubyte"))

    X_train = X_train.reshape(len(X_train), -1).astype(np.float32) / 255.0
    X_test = X_test.reshape(len(X_test), -1).astype(np.float32) / 255.0
    if args.train_limit is not None:
        X_train, y_train = X_train[:args.train_limit], y_train[:args.train_limit]
    train_mask = np.isin(y_train, [0, 1])
    test_mask = np.isin(y_test, [0, 1])

    evaluate(LogisticRegression(solver="lbfgs", max_iter=args.max_iter),
             X_train[train_mask], y_train[train_mask], X_test[test_mask], y_test[test_mask])


    options = {"solver": "lbfgs", "max_iter": args.max_iter}
    if "multi_class" in inspect.signature(LogisticRegression).parameters:
        options["multi_class"] = "multinomial"
    evaluate(LogisticRegression(**options),
             X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
