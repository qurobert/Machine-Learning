from ex03.my_linear_regression import MyLinearRegression as MyLR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def mse_(y, y_hat):
    if y.shape != y_hat.shape:
        return None
    mse = np.mean((y - y_hat) ** 2)
    return mse


# Read the data set
data = pd.read_csv("./are_blue_pills_magics.csv")
Xpill = np.array(data["Micrograms"]).reshape(-1, 1)
Yscore = np.array(data["Score"]).reshape(-1, 1)

linear_model1 = MyLR(np.array([[89.0], [-8]]))
Y_model1 = linear_model1.predict_(Xpill)

# Display the mse
print(mse_(Yscore, Y_model1))

# Plot the data set
plt.scatter(Xpill, Yscore, color='blue', label='Strue(pills)')
plt.plot(Xpill, Y_model1, color='green', label='Spredict(pills)')
plt.xlabel('Quantity of Blue Pills (in micrograms)')
plt.ylabel('Space Driving Score')
plt.legend()
plt.show()

# Plot the loss function for different values of theta1
theta1_values = np.linspace(-14, -4, 100)
theta0_values = np.linspace(80, 100, 6)  # Adjust the range as necessary
plt.figure(figsize=(10, 6))
for theta0 in theta0_values:
    loss_values = []
    for theta1 in theta1_values:
        model = MyLR(np.array([[theta0], [theta1]]))
        Y_pred = model.predict_(Xpill)
        loss = model.loss_(Yscore, Y_pred)
        loss_values.append(loss)

    plt.plot(theta1_values, loss_values, label=f'theta0 = {theta0}')

plt.xlabel('Theta1 Values')
plt.ylabel('Loss')
plt.ylim(10, 140)  # Set the y-axis to range from 20 to 140
plt.title('Evolution of the loss function J for different theta0 values')
plt.legend()
plt.show()
