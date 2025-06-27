from src.model.lenet5 import LeNet5
from src.data.loader import load_data
from src.model.optimizers import SGD
from src.utils.visualization import plot_loss
from src.utils.metrics import accuracy_score

model = LeNet5()
optimizer = SGD(lr=0.01)

X_train, y_train, X_val, y_val = load_data()

losses = []
for epoch in range(10):
    model.train(X_train, y_train)
    loss = compute_loss(...)
    losses.append(loss)
plot_loss(losses)