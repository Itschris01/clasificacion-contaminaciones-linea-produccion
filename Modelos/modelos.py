import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV

# Los 4 modelos que voy a usar para comparar
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
# Métricas para la evaluación y la Model Card
from sklearn.metrics import classification_report, accuracy_score

# Para exportar el modelo ganador
import joblib


#------------------------------------------------------------------------------------
# Cargar el dataset maestro
df = pd.read_csv("dataset_grupal_super_limpio.csv", sep=';')
X = df.drop('etiqueta', axis=1)
y = df['etiqueta']

# Dividimos: 80% para entrenar, 20% para probar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Total de imágenes: {len(df)}")
print(f"Imágenes para entrenar: {len(X_train)}")
print(f"Imágenes para probar: {len(X_test)}\n")

#------------------------------------------------------------------------------------
# Modelos que ajustan sus hiperparámetros automáticamente con GridSearchCV
#-------------------------------------------------------------------------------------

# 1. Configuración para Árbol de Decisión
# criterion: Mide la calidad de las divisiones (gini es más rápido, entropy más detallado).
# max_depth: Evita que el árbol crezca infinito y se sobreajuste (memorice).
param_grid_tree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}


# 2. Configuración para K-Nearest Neighbors (KNN)
# n_neighbors: Cuántas fotos cercanas consulta (es bueno usar impares para evitar empates).
# weights: Si los vecinos cercanos pesan más ('distance') o igual ('uniform').
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}


# 3. Configuración para Support Vector Machine (SVM)
# C: Qué tanto castigamos los errores en el entrenamiento (valores altos = más estricto).
# kernel: linear es una línea recta, rbf se adapta a formas complejas (suele ser el mejor).
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# 4. Configuración para Random Forest
# n_estimators: Número de árboles en el bosque.
# max_features: Cantidad de píxeles que cada árbol puede ver al azar.
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'criterion': ['gini', 'entropy']
}



#------------------------------------------------------------------------------------
# Entrenamiento de los modelos con búsqueda de hiperparámetros
#------------------------------------------------------------------------------------

# --- Entrenamiento de Árbol de Decisión ---
grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid_tree, cv=5)
grid_tree.fit(X_train, y_train)
print(f"Mejor Árbol: {grid_tree.best_params_} con score: {grid_tree.best_score_:.2f}")

# --- Entrenamiento de KNN ---
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
grid_knn.fit(X_train, y_train)
print(f"Mejor KNN: {grid_knn.best_params_} con score: {grid_knn.best_score_:.2f}")

# --- Entrenamiento de SVM ---
grid_svm = GridSearchCV(SVC(), param_grid_svm, cv=5)
grid_svm.fit(X_train, y_train)
print(f"Mejor SVM: {grid_svm.best_params_} con score: {grid_svm.best_score_:.2f}")

# --- Entrenamiento de Naive Bayes (Sin búsqueda de hiperparámetros) 
modelo_nb = GaussianNB()
modelo_nb.fit(X_train, y_train)
score_nb = modelo_nb.score(X_test, y_test)
print(f"Naive Bayes Score: {score_nb:.2f}\n")

# --- Entrenamiento de Random Forest ---
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5)
grid_rf.fit(X_train, y_train)
print(f"Mejor Random Forest: {grid_rf.best_params_} con score: {grid_rf.best_score_:.2f}")





#--------------------------------------------------------------------------------
# Evaluación final de los modelos en el conjunto de prueba
#--------------------------------------------------------------------------------

# Guardamos los mejores modelos encontrados en una lista para comparar
modelos_finales = {
    "Arbol": grid_tree.best_estimator_,
    "KNN": grid_knn.best_estimator_,
    "SVM": grid_svm.best_estimator_,
    "RandomForest" : grid_rf.best_estimator_,
    "NaiveBayes": modelo_nb
}

# Evaluamos cada uno en el conjunto de PRUEBA (X_test) que el modelo no ha visto
mejor_score = 0
ganador_nombre = ""
ganador_modelo = None

for nombre, modelo in modelos_finales.items():
    predicciones = modelo.predict(X_test)
    score = accuracy_score(y_test, predicciones)
    print(f"Resultado final {nombre}: {score:.4f}")
    
    if score > mejor_score:
        mejor_score = score
        ganador_nombre = nombre
        ganador_modelo = modelo

print(f"\n El modelo ganador es: {ganador_nombre} con {mejor_score:.4f} de precisión.")


#--------------------------------------------------------------------------------
# Exportación y generación de métricas para la Model Card
#--------------------------------------------------------------------------------

# 1. Exportar en formato .joblib
# Usando el formato requerido: carne_nombre_apellido.joblib
nombre_archivo_final = "C16760_Cristhian_Rojas_Alvarez.joblib" 
joblib.dump(ganador_modelo, nombre_archivo_final)
print(f" Archivo exportado exitosamente para la entrega: {nombre_archivo_final}")

# 2. Generar reporte detallado para el MODEL_CARD.md
print("\n" + "="*50)
print(f"   MÉTRICAS FINALES PARA EL MODELO: {ganador_nombre}")
print("="*50)
y_pred = ganador_modelo.predict(X_test)

# classification_report nos da Precision, Recall y F1-Score de una vez
print(classification_report(y_test, y_pred))
print("="*50)