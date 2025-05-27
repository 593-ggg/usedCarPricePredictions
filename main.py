import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import joblib


def load_data(filepath):
    """加载数据：从指定路径读取CSV文件，使用空格分隔符和GBK编码"""
    df = pd.read_csv(filepath, sep='\s+', encoding='gbk')
    return df


def clean_numeric_column(series):
    """清洗数值列：将'-'和空字符串替换为NaN，然后转换为数值类型"""
    series = series.replace('-', np.nan)
    series = series.replace('', np.nan)
    return pd.to_numeric(series, errors='coerce')


def preprocess_and_feature_engineering(df):
    # 定义需要处理的数值列
    numeric_cols = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'power',
                    'kilometer', 'regionCode', 'price'] + [f'v_{i}' for i in range(15)]

    # 清洗所有数值列
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])

    # 填充数值列的缺失值（使用中位数）
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    # 处理注册日期特征
    if 'regDate' in df.columns:
        df['regDate'] = pd.to_datetime(df['regDate'], format='%Y%m%d', errors='coerce')
        df['regYear'] = df['regDate'].dt.year
        df['regMonth'] = df['regDate'].dt.month
        df['regYear'].fillna(df['regYear'].median(), inplace=True)
        df['regMonth'].fillna(df['regMonth'].median(), inplace=True)

    # 处理创建日期特征
    if 'creatDate' in df.columns:
        df['creatDate'] = pd.to_datetime(df['creatDate'], format='%Y%m%d', errors='coerce')
        df['creatYear'] = df['creatDate'].dt.year
        df['creatMonth'] = df['creatDate'].dt.month
        df['creatYear'].fillna(df['creatYear'].median(), inplace=True)
        df['creatMonth'].fillna(df['creatMonth'].median(), inplace=True)

    # 创建车辆年龄特征（年）
    if all(col in df.columns for col in ['creatDate', 'regDate']):
        df['carAge'] = (df['creatDate'] - df['regDate']).dt.days / 365
        df['carAge'] = df['carAge'].fillna(df['carAge'].median())

    # 创建年均里程特征
    if all(col in df.columns for col in ['kilometer', 'carAge']):
        df['kmPerYear'] = df['kilometer'] / (df['carAge'].replace(0, 1))
        df['kmPerYear'] = df['kmPerYear'].fillna(df['kmPerYear'].median())

    # 创建功率里程比特征
    if all(col in df.columns for color in ['power', 'kilometer']):
        df['powerPerKm'] = df['power'] / (df['kilometer'].replace(0, 1))
        df['powerPerKm'] = df['powerPerKm'].fillna(df['powerPerKm'].median())

    # 删除原始日期列和其他无用列
    df.drop(['regDate', 'creatDate'], axis=1, errors='ignore', inplace=True)
    df.dropna(inplace=True)
    return df


def train_and_tune(X_train, y_train):
    # 只选择数值型特征
    X_train = X_train.select_dtypes(include=['int64', 'float64'])
    # 使用中位数填充缺失值
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

    # 定义参数网格（已优化过的参数）
    param_grid = {
        'n_estimators': [300],  # 树的数量
        'learning_rate': [0.12],  # 学习率
        'max_depth': [6],  # 树的最大深度
        'min_samples_split': [10],  # 分裂节点所需最小样本数
        'subsample': [0.9],  # 样本子采样比例
        'min_samples_leaf': [3]  # 叶节点最小样本数
    }

    # 初始化GBDT模型
    model = GradientBoostingRegressor(
        random_state=42,
        max_features='sqrt',
        loss='huber',
        warm_start=True
    )

    # 设置网格搜索
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=2,
        refit=True
    )

    # 执行网格搜索
    print("开始网格搜索...")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()

    # 输出结果
    print(f"网格搜索完成！总耗时: {(end_time - start_time) / 60:.1f}分钟")
    print("最佳参数:", grid_search.best_params_)
    print("最佳分数:", -grid_search.best_score_)

    # 保存最佳模型
    joblib.dump(grid_search.best_estimator_, 'best_model.pkl')
    return grid_search.best_estimator_


def main():
    """主函数：执行完整流程"""
    # 训练模型
    train_data = load_data('used_car_train_20200313.csv')
    train_processed = preprocess_and_feature_engineering(train_data)
    X = train_processed.drop(['SaleID', 'name', 'price'], axis=1)
    y = train_processed['price']  # 目标变量
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("开始训练模型...")
    best_model = train_and_tune(X_train, y_train)
    print("模型训练完成！")

    # 验证集评估
    X_val = X_val.select_dtypes(include=['int64', 'float64'])
    imputer = SimpleImputer(strategy='median')
    X_val = pd.DataFrame(imputer.fit_transform(X_val), columns=X_val.columns)
    val_pred = best_model.predict(X_val)
    mae = mean_absolute_error(y_val, val_pred)  # 计算MAE
    print(f"验证集MAE: {mae:.2f}")  # 输出评估结果

    # 测试集预测
    testB_data = load_data('used_car_testB_20200421.csv')
    testB_processed = preprocess_and_feature_engineering(testB_data)
    X_testB = testB_processed.drop(['SaleID', 'name'], axis=1)
    X_testB = X_testB.select_dtypes(include=['int64', 'float64'])

    # 填充缺失值
    imputer = SimpleImputer(strategy='median')
    X_testB = pd.DataFrame(imputer.fit_transform(X_testB), columns=X_testB.columns)

    # 预测并保存结果
    testB_pred = best_model.predict(X_testB)
    pd.DataFrame({'SaleID': testB_processed['SaleID'], 'price': testB_pred}).to_csv('submissionB.csv', index=False)


if __name__ == "__main__":
    main()