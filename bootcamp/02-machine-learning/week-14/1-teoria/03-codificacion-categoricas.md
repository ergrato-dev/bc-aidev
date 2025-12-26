# 🏷️ Codificación de Variables Categóricas

## 🎯 Objetivos

- Dominar OneHotEncoder, OrdinalEncoder y TargetEncoder
- Saber cuándo usar cada tipo de encoding
- Evitar errores comunes con LabelEncoder

---

## 📋 Contenido

### 1. El Problema de las Categóricas

Los algoritmos de ML trabajan con números. Las categorías deben convertirse:

```python
# ❌ Esto no funciona
df = pd.DataFrame({'color': ['rojo', 'verde', 'azul']})
model.fit(df, y)  # Error!

# ✅ Necesitamos encoding
df_encoded = pd.get_dummies(df)
model.fit(df_encoded, y)  # Funciona
```

![Codificación Categóricas](../0-assets/03-codificacion-categoricas.svg)

### 2. OneHotEncoder

Crea una columna binaria por cada categoría. **Sin orden implícito.**

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

df = pd.DataFrame({
    'color': ['rojo', 'verde', 'azul', 'rojo', 'verde']
})

# OneHotEncoder de sklearn
encoder = OneHotEncoder(sparse_output=False, drop=None)
encoded = encoder.fit_transform(df[['color']])

# Ver resultado
feature_names = encoder.get_feature_names_out(['color'])
df_encoded = pd.DataFrame(encoded, columns=feature_names)
print(df_encoded)
```

**Salida:**

```
   color_azul  color_rojo  color_verde
0         0.0         1.0          0.0
1         0.0         0.0          1.0
2         1.0         0.0          0.0
3         0.0         1.0          0.0
4         0.0         0.0          1.0
```

#### Opciones importantes

```python
# drop='first': Evita multicolinealidad (para regresión lineal)
encoder = OneHotEncoder(sparse_output=False, drop='first')

# handle_unknown: Maneja categorías nuevas en test
encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'  # Pone 0s en todas las columnas
)
```

**Cuándo usar:**

- ✅ Variables nominales (sin orden)
- ✅ Pocas categorías (< 10-15)
- ✅ Modelos lineales
- ❌ Alta cardinalidad (muchas categorías únicas)

### 3. OrdinalEncoder

Para variables con **orden natural**. Asigna enteros según el orden.

```python
from sklearn.preprocessing import OrdinalEncoder

df = pd.DataFrame({
    'talla': ['S', 'M', 'L', 'XL', 'M', 'S']
})

# Definir orden explícito
encoder = OrdinalEncoder(
    categories=[['S', 'M', 'L', 'XL']]  # Orden de menor a mayor
)

df['talla_encoded'] = encoder.fit_transform(df[['talla']])
print(df)
```

**Salida:**

```
  talla  talla_encoded
0     S            0.0
1     M            1.0
2     L            2.0
3    XL            3.0
4     M            1.0
5     S            0.0
```

**Cuándo usar:**

- ✅ Variables ordinales (educación, tallas, ratings)
- ✅ Tree-based models (RF, XGBoost)
- ❌ Modelos lineales (interpretan como numérico continuo)

### 4. LabelEncoder (⚠️ Solo para Target)

```python
from sklearn.preprocessing import LabelEncoder

# ✅ CORRECTO: Para variable target en clasificación
y = ['gato', 'perro', 'gato', 'pájaro']
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(y_encoded)  # [0, 2, 0, 1]

# Decodificar predicciones
y_decoded = le.inverse_transform([0, 1, 2])
print(y_decoded)  # ['gato', 'pájaro', 'perro']
```

⚠️ **NUNCA usar LabelEncoder para features**:

```python
# ❌ INCORRECTO: Introduce orden artificial
X['color_encoded'] = LabelEncoder().fit_transform(X['color'])
# azul=0, rojo=1, verde=2 → El modelo piensa rojo > azul
```

### 5. TargetEncoder (Mean Encoding)

Reemplaza categoría por la media del target para esa categoría.

```python
from sklearn.preprocessing import TargetEncoder

df = pd.DataFrame({
    'ciudad': ['Madrid', 'Barcelona', 'Madrid', 'Valencia', 'Barcelona'],
    'precio': [300000, 250000, 350000, 200000, 280000]
})

encoder = TargetEncoder(smooth='auto')
df['ciudad_encoded'] = encoder.fit_transform(
    df[['ciudad']],
    df['precio']
)
print(df)
```

**Salida:**

```
      ciudad  precio  ciudad_encoded
0    Madrid  300000          325000
1  Barcelona  250000          265000
2    Madrid  350000          325000
3  Valencia  200000          200000
4  Barcelona  280000          265000
```

**Cuándo usar:**

- ✅ Alta cardinalidad (muchas categorías)
- ✅ Cuando otras técnicas crean demasiadas columnas
- ⚠️ Usar con cross-validation para evitar leakage

```python
# Con cross-validation interno para evitar leakage
encoder = TargetEncoder(
    smooth='auto',
    target_type='continuous'  # o 'binary'
)
```

### 6. pd.get_dummies() vs OneHotEncoder

| Aspecto           | pd.get_dummies()   | OneHotEncoder |
| ----------------- | ------------------ | ------------- |
| Uso               | Exploración rápida | Producción    |
| Fit/Transform     | No                 | Sí            |
| Pipeline          | No compatible      | Compatible    |
| Nuevas categorías | Error              | Configurable  |

```python
# Exploración rápida
df_dummies = pd.get_dummies(df, columns=['color'])

# Producción (guarda el encoder)
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(X_train)
X_train_enc = encoder.transform(X_train)
X_test_enc = encoder.transform(X_test)  # Maneja categorías nuevas
```

### 7. Guía de Selección

```
¿La variable tiene orden natural?
│
├── SÍ → OrdinalEncoder
│        (tallas, educación, ratings)
│
└── NO → ¿Cuántas categorías únicas?
         │
         ├── Pocas (< 10) → OneHotEncoder
         │
         └── Muchas (≥ 10) → TargetEncoder
                            (o Feature Hashing)
```

---

## 💻 Ejemplo Completo

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Identificar columnas
nominal_cols = ['color', 'marca']
ordinal_cols = ['talla', 'calidad']

preprocessor = ColumnTransformer([
    ('nominal', OneHotEncoder(drop='first', handle_unknown='ignore'),
     nominal_cols),
    ('ordinal', OrdinalEncoder(
        categories=[['S', 'M', 'L', 'XL'], ['baja', 'media', 'alta']]
    ), ordinal_cols)
])

X_transformed = preprocessor.fit_transform(df)
```

---

## ✅ Checklist de Verificación

- [ ] Sé la diferencia entre OneHot y Ordinal
- [ ] Entiendo por qué no usar LabelEncoder en features
- [ ] Puedo aplicar TargetEncoder correctamente
- [ ] Sé elegir el encoder según el tipo de variable
