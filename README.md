<<<<<<< HEAD
# Лабораторная работа 1

**Дисциплина:** Системы искусственного интеллекта и машинное обучение  
**Тема:** Линейная регрессия и факторный анализ (PCA)  
**Вариант:** Medical Cost Personal Datasets

## Цель работы

Изучить основы линейной регрессии, построить и оценить модели на реальных данных, проанализировать мультиколлинеарность и применить PCA для снижения размерности признаков.

## Структура репозитория

- `notebooks/lab1_linear_regression_factor_analysis.ipynb` — основной отчет в Jupyter Notebook.
- `data/` — папка для данных (CSV).
- `results/` — папка для сохранения графиков и итоговых таблиц (при необходимости).
- `.gitignore` — исключения Git.
- `requirements.txt` — зависимости Python.

## Датасет

Используется датасет **Medical Cost Personal Datasets** (оригинальный источник на Kaggle):

- https://www.kaggle.com/datasets/mirichoi0218/insurance

В ноутбуке предусмотрена загрузка:
1. Из локального файла `data/insurance.csv`.
2. Если файла нет — из публичного CSV-зеркала.

## Как запустить

1. Создай и активируй виртуальное окружение:

   **Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Установи зависимости:
   ```powershell
   pip install -r requirements.txt
   ```

3. Запусти Jupyter:
   ```powershell
   jupyter notebook
   ```

4. Открой файл:
   - `notebooks/lab1_linear_regression_factor_analysis.ipynb`

## Что реализовано в ноутбуке

- Первичный анализ данных (EDA), распределения признаков и целевой переменной.
- Предобработка данных, кодирование категориальных признаков.
- Корреляционная матрица.
- Расчет VIF для проверки мультиколлинеарности.
- Модели на исходных признаках:
  - Linear Regression
  - Ridge Regression (с кросс-валидацией)
- Метрики: RMSE, R², MAPE.
- PCA после стандартизации, scree plot.
- Повторное обучение моделей на главных компонентах.
- Сравнение качества моделей до и после PCA.
- Блоки выводов для отчета.

## Список основных источников

- Документация scikit-learn: https://scikit-learn.org/stable/
- Документация statsmodels: https://www.statsmodels.org/
- Dataset (Kaggle): https://www.kaggle.com/datasets/mirichoi0218/insurance
=======
# nemchenko_ai
Лабораторные работы по предмету "Системы искусственного интеллекта и машинное обучение" от студента НГТУ Немченко Е.В. гр. АП-226
>>>>>>> fa083cf80201ad9cedc88df8e8840e91a1bb009a
