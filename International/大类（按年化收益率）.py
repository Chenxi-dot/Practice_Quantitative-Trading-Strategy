import pandas as pd
import numpy as np
from scipy.optimize import minimize

# 读取CSV文件
file_path = "E:\\大三上学科\\金融大数据分析与量化交易\\作业\\hw2\\international"
data = pd.read_csv(file_path)

# 假设列名如下（根据实际情况调整）
# 日期, 股票, 基金, 期货, 国债
columns = ["date", "股票", "基金", "期货", "国债"]
data = data[columns]

# 日期格式转换并排序
data["date"] = pd.to_datetime(data["date"])
data = data.sort_values("date")

# 计算股票、基金、期货的累计收益率和年化收益率
for asset in ["股票", "基金", "期货"]:
    data[f"{asset}_累计收益率"] = 1*((1 + data[asset]).cumprod() - 1)
    data[f"{asset}_年化收益率"] = 1*(((1 + data[f"{asset}_累计收益率"])**(222 / (data.index + 1))) - 1)

# 提取股票、基金、期货和国债的年化收益率
annualized_returns = 100*data[["股票_年化收益率", "基金_年化收益率", "期货_年化收益率"]]
annualized_returns["国债_年化收益率"] = data["国债"]  # 国债年化收益率直接使用
print(annualized_returns)

# 计算年化协方差矩阵
annualized_cov_matrix = annualized_returns.cov()

# 提取最后一行的年化收益率作为目标收益
expected_returns = annualized_returns.iloc[-1].values

# 定义马克维茨模型优化

# 目标函数：最小化组合的风险（方差）
def portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights

# 收益约束
def portfolio_return_constraint(weights, expected_returns, target_return):
    return weights.T @ expected_returns - target_return

# 权重总和约束
def weights_sum_constraint(weights):
    return np.sum(weights) - 1

# 初始权重
num_assets = len(expected_returns)
initial_weights = np.ones(num_assets) / num_assets

# 目标期望收益
target_return =13.7 # 使用平均年化收益率作为目标

# 约束和边界
constraints = [
    {'type': 'eq', 'fun': portfolio_return_constraint, 'args': (expected_returns, target_return)},
    {'type': 'eq', 'fun': weights_sum_constraint},
]
bounds = [(0, 1) for _ in range(num_assets)]

# 优化
result = minimize(
    portfolio_variance,
    initial_weights,
    args=(annualized_cov_matrix,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
)

# 输出最优配置权重
if result.success:
    optimal_weights = result.x
    assets = annualized_returns.columns
    optimal_allocation = {assets[i]: optimal_weights[i] for i in range(num_assets)}

    # 计算组合预期年化收益率和年化风险（标准差）
    portfolio_return = np.dot(optimal_weights, expected_returns)
    portfolio_variance = portfolio_variance(optimal_weights, annualized_cov_matrix)
    portfolio_std_dev = np.sqrt(portfolio_variance)

    # 夏普比率计算（使用国债的平均年化收益率作为无风险收益率）
    risk_free_rate = data["国债"].mean()  # 近似使用国债平均收益率
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev

    # 输出结果
    print("最优配置比例：", optimal_allocation)
    print("标准协方差矩阵：\n", pd.DataFrame(annualized_cov_matrix, index=assets, columns=assets))
    print(f"组合预期年化收益率: {portfolio_return:.4f}")
    print(f"组合年化方差: {portfolio_variance:.6f}")
    print(f"组合年化标准差: {portfolio_std_dev:.4f}")
    print(f"组合夏普比率: {sharpe_ratio:.4f}")
    # 导出结果到 CSV 文件
    output_file = "E:\\大三上学科\\金融大数据分析与量化交易\\作业\\hw2\\international\\optimal_portfolio_results.csv"
    results = {
        "资产类别":  ["组合预期年化收益率", "组合年化方差", "组合年化标准差", "组合夏普比率"],
        "权重或指标值":  [portfolio_return, portfolio_variance, portfolio_std_dev,
                                                             sharpe_ratio],

    }
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"结果已导出至: {output_file}")
else:
    print("优化失败：", result.message)