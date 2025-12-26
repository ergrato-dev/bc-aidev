"""
Ejercicio 01: EDA Completo - Titanic
====================================
Análisis exploratorio de datos del dataset Titanic.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
pd.set_option('display.max_columns', None)

# ============================================
# PASO 1: Cargar Datos
# ============================================
print('--- Paso 1: Cargar Datos ---')

# Descomenta las siguientes líneas:
# # Cargar desde URL de Kaggle (o descarga local)
# url_train = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
# train = pd.read_csv(url_train)
# 
# print(f'Shape: {train.shape}')
# print(f'\nColumnas: {train.columns.tolist()}')
# print(f'\nPrimeras filas:')
# print(train.head())

print()


# ============================================
# PASO 2: Información General
# ============================================
print('--- Paso 2: Información General ---')

# Descomenta las siguientes líneas:
# print('=== Info del DataFrame ===')
# print(train.info())
# 
# print('\n=== Estadísticas Descriptivas ===')
# print(train.describe())
# 
# print('\n=== Estadísticas Categóricas ===')
# print(train.describe(include=['object']))

print()


# ============================================
# PASO 3: Análisis de Missing Values
# ============================================
print('--- Paso 3: Missing Values ---')

# Descomenta las siguientes líneas:
# # Conteo de missing values
# missing = train.isnull().sum()
# missing_pct = (missing / len(train)) * 100
# 
# missing_df = pd.DataFrame({
#     'Missing': missing,
#     'Percentage': missing_pct
# }).sort_values('Missing', ascending=False)
# 
# print('=== Missing Values ===')
# print(missing_df[missing_df['Missing'] > 0])
# 
# # Visualizar missing values
# fig, ax = plt.subplots(figsize=(10, 6))
# missing_cols = missing_df[missing_df['Missing'] > 0]
# ax.barh(missing_cols.index, missing_cols['Percentage'], color='coral')
# ax.set_xlabel('Porcentaje Missing (%)')
# ax.set_title('Missing Values por Columna')
# plt.tight_layout()
# plt.show()

print()


# ============================================
# PASO 4: Distribución del Target
# ============================================
print('--- Paso 4: Distribución del Target ---')

# Descomenta las siguientes líneas:
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# 
# # Conteo
# train['Survived'].value_counts().plot(kind='bar', ax=axes[0], 
#     color=['#e74c3c', '#2ecc71'])
# axes[0].set_title('Distribución de Supervivencia')
# axes[0].set_xticklabels(['No Sobrevivió (0)', 'Sobrevivió (1)'], rotation=0)
# axes[0].set_ylabel('Cantidad')
# 
# # Porcentaje
# train['Survived'].value_counts(normalize=True).plot(kind='pie', ax=axes[1],
#     autopct='%1.1f%%', colors=['#e74c3c', '#2ecc71'],
#     labels=['No Sobrevivió', 'Sobrevivió'])
# axes[1].set_title('Porcentaje de Supervivencia')
# axes[1].set_ylabel('')
# 
# plt.tight_layout()
# plt.show()
# 
# print(f'Balance de clases:\n{train["Survived"].value_counts(normalize=True)}')

print()


# ============================================
# PASO 5: Análisis de Variables Numéricas
# ============================================
print('--- Paso 5: Variables Numéricas ---')

# Descomenta las siguientes líneas:
# numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
# 
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# axes = axes.ravel()
# 
# for ax, col in zip(axes, numeric_cols):
#     # Histograma
#     train[col].hist(ax=ax, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
#     ax.set_title(f'Distribución de {col}')
#     ax.set_xlabel(col)
#     ax.set_ylabel('Frecuencia')
#     
#     # Añadir media y mediana
#     ax.axvline(train[col].mean(), color='red', linestyle='--', label=f'Media: {train[col].mean():.2f}')
#     ax.axvline(train[col].median(), color='green', linestyle='-', label=f'Mediana: {train[col].median():.2f}')
#     ax.legend()
# 
# plt.tight_layout()
# plt.show()

print()


# ============================================
# PASO 6: Análisis de Variables Categóricas
# ============================================
print('--- Paso 6: Variables Categóricas ---')

# Descomenta las siguientes líneas:
# categorical_cols = ['Sex', 'Pclass', 'Embarked']
# 
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# 
# for ax, col in zip(axes, categorical_cols):
#     # Supervivencia por categoría
#     survival_rate = train.groupby(col)['Survived'].mean()
#     survival_rate.plot(kind='bar', ax=ax, color='teal', alpha=0.7)
#     ax.set_title(f'Tasa de Supervivencia por {col}')
#     ax.set_ylabel('Tasa de Supervivencia')
#     ax.set_ylim(0, 1)
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
#     
#     # Añadir línea de referencia
#     ax.axhline(train['Survived'].mean(), color='red', linestyle='--', 
#                label=f'Media global: {train["Survived"].mean():.2f}')
#     ax.legend()
# 
# plt.tight_layout()
# plt.show()

print()


# ============================================
# PASO 7: Análisis Bivariado
# ============================================
print('--- Paso 7: Análisis Bivariado ---')

# Descomenta las siguientes líneas:
# # Age vs Survived
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# 
# # Boxplot
# train.boxplot(column='Age', by='Survived', ax=axes[0])
# axes[0].set_title('Edad por Supervivencia')
# axes[0].set_xlabel('Survived')
# plt.suptitle('')  # Quitar título automático
# 
# # KDE plot
# train[train['Survived'] == 0]['Age'].plot(kind='kde', ax=axes[1], label='No Sobrevivió', color='red')
# train[train['Survived'] == 1]['Age'].plot(kind='kde', ax=axes[1], label='Sobrevivió', color='green')
# axes[1].set_title('Distribución de Edad por Supervivencia')
# axes[1].set_xlabel('Age')
# axes[1].legend()
# 
# plt.tight_layout()
# plt.show()
# 
# # Sex vs Pclass vs Survived
# fig, ax = plt.subplots(figsize=(10, 6))
# survival_pivot = train.pivot_table(values='Survived', index='Pclass', columns='Sex', aggfunc='mean')
# survival_pivot.plot(kind='bar', ax=ax, color=['coral', 'steelblue'])
# ax.set_title('Tasa de Supervivencia por Clase y Sexo')
# ax.set_ylabel('Tasa de Supervivencia')
# ax.set_xticklabels(['1ra Clase', '2da Clase', '3ra Clase'], rotation=0)
# ax.legend(title='Sexo')
# plt.tight_layout()
# plt.show()

print()


# ============================================
# PASO 8: Matriz de Correlaciones
# ============================================
print('--- Paso 8: Correlaciones ---')

# Descomenta las siguientes líneas:
# # Seleccionar solo numéricas
# numeric_train = train.select_dtypes(include=[np.number])
# 
# # Calcular correlación
# corr_matrix = numeric_train.corr()
# 
# # Visualizar
# plt.figure(figsize=(10, 8))
# mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
#             fmt='.2f', square=True, linewidths=0.5, mask=mask)
# plt.title('Matriz de Correlaciones')
# plt.tight_layout()
# plt.show()
# 
# # Correlación con target
# print('\n=== Correlación con Survived ===')
# print(corr_matrix['Survived'].sort_values(ascending=False))

print()


# ============================================
# PASO 9: Análisis de Outliers
# ============================================
print('--- Paso 9: Outliers ---')

# Descomenta las siguientes líneas:
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# 
# # Boxplot Age
# train.boxplot(column='Age', ax=axes[0])
# axes[0].set_title('Boxplot de Age')
# 
# # Boxplot Fare
# train.boxplot(column='Fare', ax=axes[1])
# axes[1].set_title('Boxplot de Fare')
# 
# plt.tight_layout()
# plt.show()
# 
# # Identificar outliers con IQR
# def count_outliers(series):
#     Q1 = series.quantile(0.25)
#     Q3 = series.quantile(0.75)
#     IQR = Q3 - Q1
#     outliers = ((series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)).sum()
#     return outliers
# 
# print('=== Outliers (método IQR) ===')
# for col in ['Age', 'Fare', 'SibSp', 'Parch']:
#     n_outliers = count_outliers(train[col].dropna())
#     print(f'{col}: {n_outliers} outliers')

print()


# ============================================
# PASO 10: Resumen de Insights
# ============================================
print('--- Paso 10: Resumen de Insights ---')

# Descomenta las siguientes líneas:
# print('''
# ===========================================
# RESUMEN DE INSIGHTS - EDA TITANIC
# ===========================================
# 
# 1. BALANCE DE CLASES
#    - 38% sobrevivieron, 62% no sobrevivieron
#    - Desbalance moderado
# 
# 2. MISSING VALUES
#    - Age: ~20% missing (imputar con mediana)
#    - Cabin: ~77% missing (considerar eliminar o crear flag)
#    - Embarked: <1% missing (imputar con moda)
# 
# 3. FACTORES DE SUPERVIVENCIA
#    - Sexo: Mujeres tienen 74% de supervivencia vs 19% hombres
#    - Clase: 1ra clase 63%, 2da 47%, 3ra 24%
#    - Edad: Niños tienen mayor probabilidad de supervivencia
# 
# 4. CORRELACIONES
#    - Fare correlacionado positivamente con supervivencia
#    - Pclass correlacionado negativamente (a mayor clase, menos supervivencia)
# 
# 5. FEATURES A CREAR
#    - FamilySize = SibSp + Parch + 1
#    - IsAlone = FamilySize == 1
#    - Title (extraer del nombre)
#    - AgeGroup (bins de edad)
# 
# ===========================================
# ''')

print()
print('=== Ejercicio completado ===')
