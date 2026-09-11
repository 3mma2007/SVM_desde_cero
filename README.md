# SVM desde Cero 🧠

Implementación de una **Máquina de Vectores de Soporte (SVM) lineal desde cero** usando NumPy, entrenada con descenso de gradiente sobre la función de costo *hinge loss*. El modelo se compara contra la implementación de `scikit-learn` (`SVC` con kernel lineal) sobre el dataset **Breast Cancer Wisconsin**.

## 📋 Contenido

- Carga y preparación del dataset (`load_breast_cancer` de sklearn)
- Visualización de los datos (diagrama de dispersión)
- Implementación manual de:
  - Función de costo (hinge loss + regularización L2)
  - Entrenamiento por descenso de gradiente (`fit_svm`)
  - Función de predicción (`predict_svm`)
  - Visualización de la frontera de decisión y márgenes
- Entrenamiento y evaluación del modelo manual
- Comparación con `sklearn.svm.SVC` (kernel lineal):
  - Accuracy de ambos modelos
  - Matrices de confusión lado a lado
  - Fronteras de decisión lado a lado

## 🗂️ Dataset

Se utiliza el dataset **Breast Cancer Wisconsin** incluido en scikit-learn, reducido a 2 características para poder visualizar la frontera de decisión en 2D:

- `mean radius`
- `mean texture`

Las etiquetas se transforman de `{0, 1}` a `{-1, 1}`, formato requerido por la formulación del hinge loss.

## ⚙️ Cómo funciona el modelo

El SVM se entrena minimizando la siguiente función de costo:

```
loss = (1/2) * ||w||²  +  C * mean(hinge_losses)
hinge_loss_i = max(0, 1 - yᵢ * f(xᵢ))
```

donde `f(x) = x·β` (incluyendo el sesgo). El descenso de gradiente actualiza los pesos en cada época, excluyendo el sesgo de la regularización.

> **Nota:** el hiperparámetro `C` de esta implementación **no es directamente comparable** al `C` de `sklearn.svm.SVC`, ya que sklearn usa la suma de los hinge losses en su formulación, mientras que esta implementación usa el promedio. Por eso los valores de `C` usados en ambos modelos pueden dar resultados distintos aunque sean numéricamente iguales.

## 📊 Resultados

El notebook genera:

1. **Accuracy** del modelo manual vs. sklearn sobre el conjunto de prueba (`x_test`)
2. **Matrices de confusión** comparadas lado a lado
3. **Fronteras de decisión y márgenes** comparadas lado a lado

## 🚀 Requisitos

```
numpy
pandas
matplotlib
scikit-learn
```

Instalación rápida:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## ▶️ Uso

1. Clona el repositorio
2. Abre `SVM_desde_cero.ipynb` en Jupyter Notebook, JupyterLab o Google Colab
3. Ejecuta las celdas en orden

## 📁 Estructura

```
.
├── SVM_desde_cero.ipynb   # Notebook principal
└── README.md
```

## 📚 Motivación

Este proyecto fue creado con fines educativos, para entender a fondo el funcionamiento interno de una SVM (optimización del hinge loss, regularización, gradientes) antes de usar implementaciones ya optimizadas como la de `scikit-learn`.

## 📝 Licencia

Este proyecto es de uso libre para fines educativos.
