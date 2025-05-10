# 导入必要的库
import pandas as pd  # 用于数据处理和分析
import matplotlib.pyplot as plt  # 用于数据可视化
import seaborn as sns  # 基于matplotlib的高级可视化库
from matplotlib.dates import DateFormatter  # 用于格式化日期显示
from statsmodels.tsa.arima.model import ARIMA  # 用于时间序列预测的ARIMA模型
from sklearn.ensemble import RandomForestRegressor  # 随机森林回归模型
from sklearn.metrics import mean_squared_error  # 评估模型性能的均方误差

# 读取数据
# 使用pandas读取CSV文件，并将'date'列解析为日期格式
df = pd.read_csv('DATA.csv', parse_dates=['date'])

# 设置图表样式
sns.set_style("whitegrid")  # 设置seaborn样式为白色网格背景
plt.style.use('ggplot')  # 使用ggplot样式，使图表更美观

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 股票价格走势图
def plot_stock_price_trend(instrument_code):
    """
    绘制指定股票代码的收盘价走势图
    
    参数:
        instrument_code (str): 股票代码
        
    返回:
        无返回值，直接显示图表
    """
    # 筛选指定股票的数据
    stock_df = df[df['instrument'] == instrument_code]
    
    # 创建图表
    plt.figure(figsize=(14, 7))
    # 绘制收盘价曲线
    plt.plot(stock_df['date'], stock_df['close'], label='收盘价', linewidth=2)
    # 填充曲线下方区域
    plt.fill_between(stock_df['date'], stock_df['close'], alpha=0.2)
    
    # 设置图表标题和坐标轴标签
    plt.title(f'{instrument_code} 2022年股价走势', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('价格', fontsize=12)
    # 显示网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    # 显示图例
    plt.legend()
    
    # 格式化日期显示
    date_format = DateFormatter("%Y-%m")
    plt.gca().xaxis.set_major_formatter(date_format)
    
    # 调整布局并显示图表
    plt.tight_layout()
    plt.show()

# 2. 多股票收盘价对比
def plot_multiple_stocks(instruments):
    """
    绘制多只股票的收盘价对比图
    
    参数:
        instruments (list): 包含多个股票代码的列表
        
    返回:
        无返回值，直接显示图表
    """
    plt.figure(figsize=(14, 7))
    
    # 遍历每只股票代码
    for instrument in instruments:
        stock_df = df[df['instrument'] == instrument]
        # 绘制每只股票的收盘价曲线
        plt.plot(stock_df['date'], stock_df['close'], label=instrument, linewidth=2)
    
    # 设置图表标题和坐标轴标签
    plt.title('多股票2022年收盘价对比', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('价格', fontsize=12)
    # 显示网格线和图例
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 格式化日期显示
    date_format = DateFormatter("%Y-%m")
    plt.gca().xaxis.set_major_formatter(date_format)
    
    # 调整布局并显示图表
    plt.tight_layout()
    plt.show()

# 3. 市盈率(PE)分析
def plot_pe_ratio(instrument_code):
    """
    绘制指定股票的市盈率(PE)变化图
    
    参数:
        instrument_code (str): 股票代码
        
    返回:
        无返回值，直接显示图表
    """
    stock_df = df[df['instrument'] == instrument_code]
    
    plt.figure(figsize=(14, 7))
    # 绘制PE曲线
    plt.plot(stock_df['date'], stock_df['PE_TTM'], label='市盈率(PE)', color='purple', linewidth=2)
    
    # 添加平均PE水平线
    mean_pe = stock_df['PE_TTM'].mean()
    plt.axhline(mean_pe, color='red', linestyle='--', label=f'平均PE: {mean_pe:.2f}')
    
    # 设置图表标题和坐标轴标签
    plt.title(f'{instrument_code} 2022年市盈率(PE)变化', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('市盈率', fontsize=12)
    # 显示网格线和图例
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 格式化日期显示
    date_format = DateFormatter("%Y-%m")
    plt.gca().xaxis.set_major_formatter(date_format)
    
    # 调整布局并显示图表
    plt.tight_layout()
    plt.show()

# 4. 净资产收益率(ROE)分析
def plot_roe_comparison():
    """
    绘制SSE50成分股的ROE对比柱状图
    
    返回:
        无返回值，直接显示图表
    """
    # 获取每只股票的最新ROE值
    latest_roe = df.sort_values('date').groupby('instrument').last()['ROE']
    
    plt.figure(figsize=(12, 6))
    # 绘制ROE降序排列的柱状图
    latest_roe.sort_values(ascending=False).plot(kind='bar', color='teal')
    
    # 设置图表标题和坐标轴标签
    plt.title('SSE50成分股净资产收益率(ROE)对比', fontsize=16)
    plt.xlabel('股票代码', fontsize=12)
    plt.ylabel('ROE', fontsize=12)
    # 旋转x轴标签
    plt.xticks(rotation=45)
    # 显示y轴网格线
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 调整布局并显示图表
    plt.tight_layout()
    plt.show()

# 5. 交易量分析
def plot_volume_analysis(instrument_code):
    """
    绘制指定股票的价格和交易量分析图
    
    参数:
        instrument_code (str): 股票代码
        
    返回:
        无返回值，直接显示图表
    """
    stock_df = df[df['instrument'] == instrument_code]
    
    # 创建包含两个子图的图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # 在上方子图绘制收盘价曲线
    ax1.plot(stock_df['date'], stock_df['close'], label='收盘价', color='blue')
    ax1.set_ylabel('价格', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 在下方子图绘制交易量柱状图
    ax2.bar(stock_df['date'], stock_df['amount'], label='交易量', color='orange', alpha=0.7)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_ylabel('交易量', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 格式化日期显示
    date_format = DateFormatter("%Y-%m")
    ax2.xaxis.set_major_formatter(date_format)
    
    # 设置总标题并调整布局
    plt.suptitle(f'{instrument_code} 2022年价格与交易量分析', fontsize=16)
    plt.tight_layout()
    plt.show()

# 6. 相关性热力图
def plot_correlation_heatmap(instrument_code):
    """
    绘制指定股票的财务指标相关性热力图
    
    参数:
        instrument_code (str): 股票代码
        
    返回:
        无返回值，直接显示图表
    """
    stock_df = df[df['instrument'] == instrument_code]
    # 选择要分析的数值列
    numeric_cols = ['ROE', 'Free_Cash_Flow', 'EPS', 'Net_Profit_Qoq', 'PE_TTM', 'Market_Cap', 'return_5', 'amount']
    
    # 计算各指标间的相关系数
    corr = stock_df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    # 绘制热力图
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(f'{instrument_code} 财务指标相关性热力图', fontsize=16)
    plt.tight_layout()
    plt.show()

# 7. ARIMA时间序列预测
def arima_forecast(instrument_code, days=30):
    """
    使用ARIMA模型预测股票未来价格
    
    参数:
        instrument_code (str): 股票代码
        days (int): 预测天数，默认为30天
        
    返回:
        pandas.Series: 包含预测结果的序列
    """
    stock_df = df[df['instrument'] == instrument_code].sort_values('date')
    # 创建ARIMA模型(5,1,0)
    model = ARIMA(stock_df['close'], order=(5,1,0))
    # 拟合模型
    model_fit = model.fit()
    # 进行预测
    forecast = model_fit.get_forecast(steps=days)
    return forecast.predicted_mean

# 8. 随机森林回归预测
def random_forest_predict(instrument_code):
    """
    使用随机森林回归模型预测股票价格
    
    参数:
        instrument_code (str): 股票代码
        
    返回:
        tuple: (预测结果数组, 均方误差)
    """
    stock_df = df[df['instrument'] == instrument_code].sort_values('date')
    # 选择特征列
    X = stock_df[['PE_TTM', 'ROE', 'EPS', 'amount']]
    # 目标变量
    y = stock_df['close']
    # 创建随机森林回归模型
    model = RandomForestRegressor(n_estimators=100)
    # 使用前N-30天数据训练模型
    model.fit(X[:-30], y[:-30])
    # 预测最后30天的价格
    predictions = model.predict(X[-30:])
    # 计算均方误差
    mse = mean_squared_error(y[-30:], predictions)
    return predictions, mse

# 调用各个函数进行可视化分析
plot_stock_price_trend('00320.SHA')
plot_multiple_stocks(['00320.SHA', '10320.SHA'])
plot_pe_ratio('00320.SHA')
plot_roe_comparison()
plot_volume_analysis('00320.SHA')
plot_correlation_heatmap('00320.SHA')

# 打印预测结果
print("ARIMA 30天预测结果:", arima_forecast('00320.SHA'))
rf_pred, rf_mse = random_forest_predict('00320.SHA')
print("随机森林预测MSE:", rf_mse)