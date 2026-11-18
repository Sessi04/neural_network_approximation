import numpy as np
import matplotlib . pyplot as plt
from keras . models import Sequential
from keras . layers import Dense
from keras . optimizers import Adam
# 1. Données d’entrainement
N = 1000
x = np. linspace (0, 2 * np.pi , N)
y_true = np.sin(x)
x_train = x. reshape (-1, 1)
y_train = y_true . reshape (-1, 1)
# 2. Réseau de neurones
model = Sequential ()
model.add( Dense (10 , activation ='sigmoid', input_shape =(1 ,)))
model.add( Dense (1, activation ='linear'))
# 3. Compilation avec Adam ( meilleur pour convergence )
model. compile ( optimizer =Adam( learning_rate =0.01) , loss='mse')
# 4. Entrainement
model.fit(x_train , y_train , epochs =1000 , verbose =0)
# 5. Prédiction
y_pred = model . predict ( x_train ). flatten ()
# 6. Affichage
plt. figure ( figsize =(10 , 6))
plt.plot(x, y_true , label ='sin(x)', color ='blue', linewidth =2)
plt.plot(x, y_pred , label ='Approximation', color ='red', linewidth =1)
plt. legend ()
plt.title('Approximation precise de sin(x)')
plt.grid(True)
plt.show ()
from sklearn . metrics import mean_squared_error
rmse = np.sqrt( mean_squared_error (y_true , y_pred ))
print (f"RMSE = {rmse :.6f}")