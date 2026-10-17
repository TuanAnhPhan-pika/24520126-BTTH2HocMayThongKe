import numpy as np
import idx2numpy
from matplotlib import pyplot as plt
from pathlib import Path

from sklearn.metrics import classification_report

from logistic_regression import LogisticRegression

data_dir = Path(__file__).resolve().parent / "data"

train_images = idx2numpy.convert_from_file(str(data_dir / "train-images.idx3-ubyte"))
train_labels = idx2numpy.convert_from_file(str(data_dir / "train-labels.idx1-ubyte"))

test_images = idx2numpy.convert_from_file(str(data_dir / "t10k-images.idx3-ubyte"))
test_labels = idx2numpy.convert_from_file(str(data_dir / "t10k-labels.idx1-ubyte"))

train_mask = (train_labels == 1) | (train_labels == 0)
test_mask = (test_labels == 1) | (test_labels == 0)
train_images = train_images[train_mask]
train_labels = train_labels[train_mask]
test_images = test_images[test_mask]
test_labels = test_labels[test_mask]

N, *_ = train_images.shape
train_images = train_images.reshape(N, -1)

N, *_ = test_images.shape
test_images = test_images.reshape(N, -1)

train_images = train_images.astype(np.float64) / 255.0
test_images = test_images.astype(np.float64) / 255.0

model = LogisticRegression(
    epoch = 100,
    lr = 0.1,)

model.fit(train_images, train_labels)

predicted_labels = model.predict(test_images)

metrics = model.evaluate(
    test_labels,
    predicted_labels
)

print("\nClassification Report:")
print(
    classification_report(
        test_labels,
        predicted_labels,
        digits = 4
    )
)
