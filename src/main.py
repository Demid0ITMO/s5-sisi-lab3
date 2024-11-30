import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ДАТАСЕТ
ds = pd.read_csv("california_housing_train.csv")

# СТАТИСТИКА ПО ДАТАСЕТУ
stats = ds.describe()

# КОЛИЧЕСТВО, MIN, MAX
print(stats.loc['count'])
print(stats.loc['min'])
print(stats.loc['max'])

# СРЕДНИЕ ЗНАЧЕНИЯ
plt.figure(figsize=(6, 3))
plt.plot(stats.loc['mean'].index, stats.loc['mean'], label='Mean', marker='o')
plt.title('Mean vals of columns', fontsize=8)
plt.xlabel('Columns', fontsize=8)
plt.ylabel('Mean Values', fontsize=8)
plt.legend()
plt.xticks(rotation=15, fontsize=8)
plt.tight_layout()
plt.show()

# СРЕДНИЕ ЗНАЧЕНИЯ
plt.figure(figsize=(6, 3))
plt.plot(stats.loc['std'].index, stats.loc['std'], label='Std Dev', marker='o')
plt.title('Standard Deviation of Numerical Columns', fontsize=8)
plt.xlabel('Columns', fontsize=8)
plt.ylabel('Standard Deviation', fontsize=8)
plt.legend()
plt.xticks(rotation=15, fontsize=8)
plt.tight_layout()
plt.show()

# КВАНТИЛИ
quantiles = stats.loc[['25%', '50%', '75%']]
for quantile in quantiles.index:
    plt.figure(figsize=(6, 3))
    plt.plot(quantiles.loc[quantile].index, quantiles.loc[quantile], label=f'{quantile} Quantile', marker='o')
    plt.title(f'{quantile} Quantile of Numerical Columns', fontsize=8)
    plt.xlabel('Columns', fontsize=8)
    plt.ylabel(f'{quantile} Quantile Values', fontsize=8)
    plt.legend()
    plt.xticks(rotation=15, fontsize=8)
    plt.tight_layout()
    plt.show()

# ПРЕДОБРАБОТКА

# проверим есть ли вообще пустые значения
missing_values = ds.isna()
missing_counts = missing_values.sum()
print(missing_counts)
# пустых значений нет, кайфуем

# нормирование
ds = (ds - ds.min()) / (ds.max() - ds.min())


# РАЗДЕЛЕНИЕ НА ТЕСТОВЫЙ И ОБУЧАЮЩИЙ НАБОР
test_size = 0.2
test_rows = int(len(ds) * test_size)
ds_shuffled = ds.sample(frac=1, random_state=17).reset_index(drop=True)
train_data = ds_shuffled.iloc[test_rows:]
test_data = ds_shuffled.iloc[:test_rows]


# Разделение признаков и целевой переменной
target = 'median_house_value'
priznaki_train = train_data.drop(columns=[target])
target_train = train_data[target].values
priznaki_test = test_data.drop(columns=[target])
target_test = test_data[target].values

# 3 набора параметров
params = [
    ['housing_median_age', 'total_rooms', 'total_bedrooms'],
    ['population', 'households'],
    ['longitude', 'latitude']
]


def mnk(x, y):
    # свободный член
    x_b = np.c_[np.ones((x.shape[0], 1)), x]
    # коэффициенты по формуле (X^T X)^-1 X^T y
    coef = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
    return coef


# Выбор признаков для каждой модели
x_trains = list(map(lambda p: train_data[p].values, params))

# коэффиценты по МНК
for i in range(len(x_trains)):
    print(f"Коэффициенты мнк для модели {i+1}:", mnk(x_trains[i], target_train))


def predict(x, coefficients):
    x_b = np.c_[np.ones((x.shape[0], 1)), x]
    predictions = x_b.dot(coefficients)
    return predictions


x_tests = list(map(lambda p: test_data[p].values, params))

predicts = []
for i in range(len(x_tests)):
    predicts.append(predict(x_tests[i], mnk(x_trains[i], target_train)))


def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2


r2_scores = []
for i in range(len(predicts)):
    r2_scores.append(r2_score(target_test, predicts[i]))
    print(f'Коэффициент детерминации для модели {i+1}:', r2_scores[i])
