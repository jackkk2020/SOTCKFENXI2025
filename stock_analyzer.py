import streamlit as st
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import matplotlib.pyplot
from datetime import datetime, relativedelta

# 设置全局样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统支持的字体
plt.rcParams['axes.unicode_minus'] = False

# 自定义双色系
dual_colors = ['#4CB050', '#FF3333']  # 绿跌红涨

# 数据获取
@st.cache_data(ttl=3600)
def get_stock_data(symbol):
    try:
        # 获取当前日期
        end_date = datetime.now().strftime("%Y%m%d")
        # 获取个股历史数据
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date="20110101", end_date=end_date)
        if df.empty:
            return DataFrame()
            
        df = df.rename(columns={
            '日期': 'date',
            '收盘': 'close',
            '涨跌幅': 'change'
        })
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        return df.sort_values('date')
    except Exception as e:
        st.error(f"数据获取失败：{str(e)}")
        return pd.DataFrame()

# 构建月度数据
def build_monthly_table(df):
    monthly = []
    current = datetime.now().year
    current_month = datetime.now().month
    
    for y in range(2015, current_year + 1):
        row = {'年份': y}
        for m in range(1, 13):
            if y == current_year and m > current_month:
                break
                
            sub = df[(df['year'] == y) & (df['month'] == m)]
            if len(sub) >= 3:  # 至少3个交易日才计算
                try:
                    chg = (sub.iloc[-1]['close'] / sub.iloc['close'] - 1) * 100
                    row[f'{m}月'] = round(chg, 2)
                except:
                    row[f'{m}月'] = None
            else:
                row[f'{m}月'] = None
        monthly.append(row)
    return pd.DataFrame(monthly).set_index('年份')

# 计算近6个月高低点
def get_recent_high_low(df, months=6):
    end_date = datetime.now()
    start_date = end_date - relativedelta(months=months)
    recent_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    if not recent_data.empty:
        return recent_data['close'].max(), recent_data['close'].min()
    return None, None

# 计算月度涨跌概率
def calculate_monthly_probability(df):
    prob_data = []
    for m in range(1, 13):
        month_col = f'{m}月'
        if month_col in df.columns:
            month_data = df[month_col].dropna()
            if len(month_data) > 0:
                up_prob = (month_data >= 0).mean() * 100
                down_prob = (month_data < 0).mean() * 100
                avg_change = month_data.mean()
                max_up = month_data.max()
                max_down = month_data.min()
                prob_data.append({
                    '月份': f'{m}月',
                    '上涨概率(%)': round(up_prob, 1),
                    '下跌概率(%)': round(down_prob, 1),
                    '平均涨跌幅(%)': round(avg_change, 2),
                    '最大涨幅(%)': round(max_up, 2),
                    '最大跌幅(%)': round(max_down, 2),
                    '上涨月份数': f"{sum(month_data >= 0)}/{len(month_data)}"
                })
    return pd.DataFrame(prob_data).set_index('月份')

# 获取股票名称
def get_stock_name(symbol):
    try:
        stock_info = ak.stock_individual_info_em(symbol=symbol)
        return stock_info.loc[stock_info['item'] == '股票简称', 'value'].values[0]
    except:
        return f"未知股票({symbol})"

def app():
    st.title("📈 A股个股月度数据分析工具")
    
    # 股票代码输入
    stock_code = st.text_input(
        "输入A股股票代码(如:600519)", 
        value="600519",
        help="输入6位数字的A股股票代码，SH/SZ后缀可选"
    ).strip()
    
    # 清空输入
    stock_code = ''.join(filter(str.isdigit, stock_code))[:6]
    if not stock_code:
        st.warning("请输入有效的股票代码")
        return
        
    # 获取数据
    with st.spinner(f"正在获取 {stock_code} 的历史数据..."):
        df = get_stock_data(stock_code)
        if df.empty:
            st.error("未能获取该股票数据，请检查代码是否正确或稍后再试")
            return
            
    stock_name = get_stock_name(stock_code)
    
    # 计算各类指标
    latest_close = df.iloc[-1]['close']
    hist_high = df['close'].max()
    recent_high, recent_low = get_recent_high_low(df)
    valid_months = monthly_df.notna().sum().sum()
    
    # 计的计算
    if recent_high and recent_low:
        range_pct = ((recent_high - recent_low) / recent_low) * 100
        from_high = ((1 - (latest_close / recent_high)) * 100)
        from_low = ((latest_close / recent_low - 1) * 100)
        position_ratio = ((latest_close - recent_low) / (recent_high - recent_low)) * 100
    else:
        range_pct = from_high = from_low = position_ratio = "N/A"
    
    # 显示股票信息
    st.markdown(f"""
    ### 🏷️ {stock_name}({stock_code})
    ▸ **数据范围**: 2015年1月 - {datetime.now().strftime('%Y年%m月')}
    ▸ **最后更新**: {df['date'].max().strftime('%Y-%m-%d')}
    """)
    
    # 关键指标卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "最新收盘价", 
            f"{latest_close:.2f}",
            help="当前最新收盘价格"
        )
    with col2:
        st.metric(
            "历史最高价", 
            f"{hist_high:.2f}", 
            delta=f"差:{((latest_close/hist_high-1)*100):.1f}%",
            help="历史最高收盘价及当前差距"
        )
    with col3:
        st.metric(
            "6个月区间", 
            f"{recent_low:.2f}-{recent_high:.2f}",
            delta=f"振幅:{range_pct:.1f}%",
            help="近6个月最低价-最高价及波动幅度"
        )
    with col4:
        st.metric(
            "区间位置", 
            position_ratio,
            delta=f"↑{from_low} ↓{from_high}",
            help="当前价在6个月区间中的位置(0%-100%)"
        )
    
    # 原始月度数据
    st.markdown("---")
    st.subheader("📅 详细月度数据")
    styled_monthly = monthly_df.style.format('{:.1f}%', na_rep="-").applymap(
        lambda val: f'background-color: {dual_colors[1] if val is not None and val >= 0 else dual_colors[0]}; color: white'
    )
    st.dataframe(
        styled_monthly,
        height=600,
        use_container_width=True
    )

if __name__ == "__main__":
    app()
