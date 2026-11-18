# === IMPORTATION DES LIBRAIRIES ===

# numpy : Pour les calculs numériques et manipulations de tableaux
import numpy as np

# matplotlib : Pour créer des visualisations
import matplotlib . pyplot as plt

# Axes3D : Pour les graphiques en 3D
from mpl_toolkits . mplot3d import Axes3D

# train_test_split : Pour diviser les données en ensembles d’entraînement et de test
from sklearn . model_selection import train_test_split

# MinMaxScaler : Pour normaliser les données
from sklearn . preprocessing import MinMaxScaler

# Sequential : Type de modèle de réseau de neurones ( couches empilées)
from tensorflow . keras . models import Sequential

# Dense : Couche de neurones complètement connectée
# BatchNormalization : Normalise les activations entre les couches
# Dropout : Technique de régularisation
from tensorflow . keras . layers import Dense , BatchNormalization , Dropout

# Adam : Optimiseur pour l’entraînement
from tensorflow . keras . optimizers import Adam

# EarlyStopping : Arrête l’entraînement si plus d’amélioration
# LearningRateScheduler : Permet de modifier le taux d’apprentissage pendant l’entraînement
from tensorflow . keras . callbacks import EarlyStopping , LearningRateScheduler

# l2 : Ré gularisation L2 pour é viter le surapprentissage
from tensorflow . keras . regularizers import l2

# swish : Fonction d’activation moderne souvent plus performante que
ReLU
from tensorflow . keras . activations import swish


# === DÉ FINITION DE LA FONCTION ROSENBROCK ===
def rosenbrock (x1 , x2 , a=1, b =100) :
#Fonction mathématique utilisée comme benchmark en optimisation
#Minimum global à (1 ,1) , difficile à approximer à cause de sa vallée
#parabolique
#Formule : f(x1 ,x2) = (a-x1)^2 + b*(x2 -x1 ^2)^2
   return (a - x1)**2 + b * (x2 - x1 **2) **2

# === GÉNÉRATION DES DONNÉES ===

# Nombre total de points de données
N = 15000

# Création des points x1 :
# - 60% de points près du minimum ( distribution normale centrée sur 1)
# - 40% de points répartis uniformément pour explorer l’espace
x1 = np. concatenate (
   [np. random . normal (1, 0.2 , int(N *0.6) ), # Points concentrés près du minimum
    np. random . uniform (-2, 2, int(N *0.4) )
    # Points répartis uniformément
    ])

# Même distribution pour x2
x2 = np. concatenate ([np. random . normal (1, 0.2 , int(N *0.6) ),np. random . uniform (-1, 3, int(N *0.4) )])

# Ajout de points spécifiques le long de la vallée parabolique (x2 =x1 ^2)

valley_samples = int(N *0.2) # 20% de points suppl é mentaires
x1_valley = np. random . uniform ( -1.5 , 1.5 , valley_samples )
x2_valley = x1_valley **2 + np. random . normal (0, 0.1 , valley_samples )

# Combinaison de tous les points
x1 = np. concatenate ([x1 , x1_valley ])
x2 = np. concatenate ([x2 , x2_valley ])

# Création de la matrice d’entrée X ( shape : Nx2)
X = np. column_stack ((x1 , x2))

# Calcul des valeurs y correspondantes ( shape : Nx1)
y = rosenbrock (x1 , x2). reshape (-1, 1)

# === PRÉPARATION DES DONNÉES ===

# Normalisation des entrées entre -1 et 1
scaler_X = MinMaxScaler ( feature_range =(-1, 1))
X_scaled = scaler_X . fit_transform (X)

# Normalisation des sorties entre 0 et 1
scaler_y = MinMaxScaler ()
y_scaled = scaler_y . fit_transform (y)

# Division des données en ensembles d’entraînement (85%) et de test(15%)
X_train , X_test , y_train , y_test = train_test_split (
X_scaled , y_scaled ,
test_size =0.15 , # 15% pour le test
random_state =42 # Pour reproductibilité
)

# === CONSTRUCTION DU MODÈLE ===

model = Sequential ([ # Modèle séquentiel ( couches empilées). Première couche cachée avec 512 neurones
Dense (512 , activation =swish , input_shape =(2 ,) , kernel_regularizer =l2 (1e -4)),
BatchNormalization () , # Normalise les sorties de la couche
Dropout (0.1) , # Désactive aléatoirement 10% des neurones

# Deuxième couche cachée avec 256 neurones
Dense (256 , activation =swish , kernel_regularizer =l2 (1e -4)),
BatchNormalization () ,

# Troisième couche cachée avec 128 neurones
Dense (128 , activation =swish , kernel_regularizer =l2 (1e -4)),
BatchNormalization () ,

# Quatrième couche cachée avec 64 neurones
Dense (64 , activation = swish ),
BatchNormalization () ,

# Cinquième couche cachée avec 32 neurones
Dense (32 , activation = swish ),

# Couche de sortie avec 1 neurone ( activation linéaire pour la régression )
Dense (1, activation ='linear')])

# === CONFIGURATION DE L’OPTIMISATION ===

def lr_scheduler (epoch , lr):
#""" Fonction pour ré duire le learning rate pendant l’entraînement """
    if epoch % 30 < 15: # Pendant les 15 premières epochs de chaque cycle de 30
        return lr # Garde le même learning rate
    else:
        return lr * 0.8 # Réduit le learning rate de 20%

# Configuration de l’optimiseur Adam
optimizer = Adam(
learning_rate =0.0005 , # Taux d’apprentissage initial
beta_1 =0.9 , # Param ètre pour la moyenne du gradient
beta_2 =0.999 , # Param ètre pour la moyenne des carrés du gradient
epsilon =1e -7 # Terme de stabilit é numé rique
)

# Compilation du modèle
model. compile (
optimizer =optimizer ,
loss='mse', # Fonction de perte : erreur quadratique moyenne
metrics =[ 'mae ', 'mse' ] # Métriques à suivre pendant l’entraînement
                )

# === ENTRAÎNEMENT DU MODÈLE ===

# Configuration de l’early stopping
early_stop = EarlyStopping (
monitor ='val_mse ', # Surveille l’erreur de validation
patience =40 , # Nombre d’epochs sans amélioration avant arrêt
min_delta =1e-6, # Amélioration minimale requise
restore_best_weights =True # Garde les meilleurs poids trouvés
)

# Lancement de l’entraînement
history = model .fit(
X_train , y_train ,
validation_split =0.15 , # Utilise 15% des données d’entraînement pour la validation
epochs =800 , # Nombre maximum d’itérations
batch_size =256 , # Taille des lots
callbacks =[ early_stop , LearningRateScheduler ( lr_scheduler )], #Callbacks
verbose =1 # Affiche la progression
)

# === ÉVALUATION DU MODÈLE ===

# Calcul des métriques sur l’ensemble de test
test_loss , test_mae , test_mse = model . evaluate (X_test , y_test , verbose=0)

# Affichage des résultats
print (f"\ nPerformance finale :")
print (f"- Test MSE: { test_mse :.2e}")
print (f"- Test MAE: { test_mae :.2e}")
# Erreur quadratique moyenne
# Erreur absolue moyenne

# === PRÉDICTION SUR UNE GRILLE ===

# Création d’une grille fine pour la visualisation
x1_grid = np. linspace (-2, 2, 300) # 300 points entre -2 et 2
x2_grid = np. linspace (-1, 3, 300) # 300 points entre -1 et 3
X1 , X2 = np. meshgrid (x1_grid , x2_grid ) # Crée une grille 2D

# Transformation de la grille pour la prédiction
X_grid = scaler_X . transform (np. column_stack (( X1. ravel () , X2. ravel ())))

# Prédiction par batch pour économiser la mémoire
batch_size = 10000
y_pred = np. zeros (X1. ravel (). shape [0]) # Initialise un tableau vide pour les prédictions

for i in range (0, X_grid . shape [0] , batch_size ):
    batch = X_grid [i:i+ batch_size ] # Prend un sous - ensemble de la grille
# Fait la prédiction et la stocke
    y_pred [i:i+ batch_size ] = scaler_y . inverse_transform (
    model. predict (batch , verbose =0)). flatten ()

# Remet les prédictions dans la forme de la grille
y_pred = y_pred . reshape (X1. shape )

# === VISUALISATION 3D ===

fig = plt. figure ( figsize =(22 , 7)) # Crée une figure large
fig. suptitle (" Approximation haute précision de la fonction de Rosenbrock ", y=1.02 , fontsize =16)
# Graphique 1: Surface théorique
ax1 = fig. add_subplot (131 , projection ='3d')
surf1 = ax1. plot_surface (X1 , X2 , rosenbrock (X1 , X2), cmap='viridis', alpha =0.9 , edgecolor ='none')
ax1. set_title (" Surface théorique ", pad =15 , fontsize =14)
ax1. set_xlabel ("x1", labelpad =12 , fontsize =12)
ax1. set_ylabel ("x2", labelpad =12 , fontsize =12)
ax1. set_zlabel ("f(x1 ,x2)", labelpad =12 , fontsize =12)
ax1. view_init (elev =30 , azim =45) # Angle de vue
fig. colorbar (surf1 , ax=ax1 , shrink =0.6 , pad =0.1) # Ajoute une barre de couleur

# Graphique 2: Surface prédite par le réseau
ax2 = fig. add_subplot (132 , projection ='3d')
surf2 = ax2. plot_surface (X1 , X2 , y_pred ,
cmap='plasma', alpha =0.9 , edgecolor ='none')
ax2. set_title (" Approximation du réseau", pad =15 , fontsize =14)
ax2. set_xlabel ("x1", labelpad =12 , fontsize =12)
ax2. set_ylabel ("x2", labelpad =12 , fontsize =12)
ax2. view_init (elev =30 , azim =45)
fig. colorbar (surf2 , ax=ax2 , shrink =0.6 , pad =0.1)

# Graphique 3: Erreur relative entre théorie et prédiction
ax3 = fig. add_subplot (133 , projection = '3d')
true_vals = rosenbrock (X1 , X2) # Valeurs théoriques
error = np.abs( true_vals - y_pred ) / (np.abs( true_vals ) + 1e -6)

#Calcul erreur relative
surf3 = ax3. plot_surface (X1 , X2 , error ,
cmap='hot', alpha =0.9 , edgecolor ='none')
ax3. set_title (" Erreur relative (%)", pad =15 , fontsize =14)
ax3. set_xlabel ("x1", labelpad =12 , fontsize =12)
ax3. set_ylabel ("x2", labelpad =12 , fontsize =12)
ax3. view_init (elev =30 , azim =45)
fig. colorbar (surf3 , ax=ax3 , shrink =0.6 , pad =0.1 , format ='%.0e')

#Format scientifique
plt. tight_layout () # Améliore l’espacement
plt. savefig (" high_precision_rosenbrock .png", dpi =300 , bbox_inches ='tight' ) # Sauvegarde
plt.show () # Affiche

# === COURBES D’APPRENTISSAGE ===
plt. figure ( figsize =(12 , 6))
# Affiche l’erreur d’entraînement (échelle logarithmique )
plt. semilogy ( history . history ['loss'], label ='Train Loss', linewidth =2)
# Affiche l’erreur de validation (échelle logarithmique )
plt. semilogy ( history . history ['val_loss'], label ='Validation Loss',linewidth =2)
plt.title (" Courbes d’apprentissage (échelle logarithmique )", fontsize=14)
plt. xlabel (" Epochs ", fontsize =12)
plt. ylabel ("MSE", fontsize =12)
plt. legend ( fontsize =12) # Affiche la lé gende
plt.grid(True , which ="both", linestyle ='--', alpha =0.7) # Grille
plt. savefig (" improved_learning_curves .png", dpi =300 , bbox_inches ='tight')
plt.show ()

# === VISUALISATION 2D ( COUPES ) ===
plt. figure ( figsize =(15 , 6))
# Crée une figure large
# Coupe longitudinale (x2 = 1)
plt. subplot (1, 2, 1) # Premier sous - graphique
x1_line = np. linspace (-2, 2, 500) # 500 points entre -2 et 2
x2_line = np. ones_like ( x1_line ) # x2 fixé à 1
X_line = scaler_X . transform (np. column_stack (( x1_line , x2_line ))) #Normalisation
y_pred_line = scaler_y . inverse_transform ( model . predict (X_line , verbose=0)) # Prédiction
plt.plot(x1_line , rosenbrock (x1_line , x2_line ), label ='Vraie fonction', linewidth =3)
plt.plot(x1_line , y_pred_line , '--', label ='Approximation', linewidth=3)
plt.title (" Coupe longitudinale (x2 = 1)", fontsize =14)
plt. xlabel ("x1", fontsize =12)
plt. ylabel ("f(x1 , x2)", fontsize =12)
plt. legend ( fontsize =12)
plt.grid(True)
# Coupe transversale (x1 = 1)
plt. subplot (1, 2, 2) # Deuxième sous-graphique
x2_line = np. linspace (-1, 3, 500) # 500 points entre -1 et 3
x1_line = np. ones_like ( x2_line ) # x1 fixé à 1
X_line = scaler_X . transform (np. column_stack (( x1_line , x2_line ))) #Normalisation
y_pred_line = scaler_y . inverse_transform ( model . predict (X_line , verbose=0)) # Prédiction
plt.plot(x2_line , rosenbrock (x1_line , x2_line ), label ='Vraie fonction', linewidth =3)
plt.plot(x2_line , y_pred_line , '--', label ='Approximation', linewidth=3)
plt.title (" Coupe transversale (x1 = 1)", fontsize =14)
plt. xlabel ("x2", fontsize =12)
plt. ylabel ("f(x1 , x2)", fontsize =12)
plt. legend ( fontsize =12)
plt.grid(True)
plt. tight_layout () # Améliore l’espacement
plt. savefig (" cross_sections .png", dpi =300 , bbox_inches ='tight')#Sauvegarde
plt.show () # Affiche