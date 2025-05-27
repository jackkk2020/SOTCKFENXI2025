import streamlit as st
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import os
import requests
import logging
from dotenv import load_dotenv

# 设置全局样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统支持的字体
plt.rcParams['axes.unicode_minus'] = False

# 自定义双色系
dual_colors = ['#4CB050', '#FF3333']  # 绿跌红涨

class StockAnalyzer:
    def __init__(self):
        # 设置日志
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # 加载环境变量
        load_dotenv()

        # 配置参数 - 调整为适合周线的参数
        self.params = {
            'ma_periods': {'short': 4, 'medium': 12, 'long': 24},  # 4 周(1 个月), 12 周(3 个月), 24 周(6 个月)
            'rsi_period': 12,  # 12 周 RSI
            'bollinger_period': 20,  # 20 周布林带
            'bollinger_std': 2,
            'volume_ma_period': 12,  # 12 周均量线
            'atr_period': 14  # 14 周 ATR
        }

    def get_weekly_data(self, df):
        """将日线数据转换为周线数据"""
        try:
            # 确保日期格式正确
            df['date'] = pd.to_datetime(df['date'])

            # 转换为周线数据
            df = df.set_index('date')
            weekly_df = df.resample('W').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            return weekly_df.reset_index().sort_values('date')
        except Exception as e:
            self.logger.error(f"转换周线数据失败: {str(e)}")
            raise Exception(f"转换周线数据失败: {str(e)}")

    def calculate_ema(self, series, period):
        """计算指数移动平均线"""
        return series.ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, series, period):
        """计算 RSI 指标"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, series):
        """计算 MACD 指标 - 调整为周线参数"""
        exp1 = series.ewm(span=6, adjust=False).mean()  # 6 周 EMA
        exp2 = series.ewm(span=13, adjust=False).mean()  # 13 周 EMA
        macd = exp1 - exp2
        signal = macd.ewm(span=5, adjust=False).mean()  # 5 周信号线
        hist = macd - signal
        return macd, signal, hist

    def calculate_bollinger_bands(self, series, period, std_dev):
        """计算布林带"""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    def calculate_atr(self, df, period):
        """计算 ATR 指标"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def calculate_indicators(self, df):
        """计算技术指标"""
        try:
            # 计算移动平均线
            df['MA4'] = self.calculate_ema(df['close'], self.params['ma_periods']['short'])
            df['MA12'] = self.calculate_ema(df['close'], self.params['ma_periods']['medium'])
            df['MA24'] = self.calculate_ema(df['close'], self.params['ma_periods']['long'])

            # 计算 RSI
            df['RSI'] = self.calculate_rsi(df['close'], self.params['rsi_period'])

            # 计算 MACD
            df['MACD'], df['Signal'], df['MACD_hist'] = self.calculate_macd(df['close'])

            # 计算布林带
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = self.calculate_bollinger_bands(
                df['close'],
                self.params['bollinger_period'],
                self.params['bollinger_std']
            )

            # 成交量分析
            df['Volume_MA'] = df['volume'].rolling(window=self.params['volume_ma_period']).mean()
            df['Volume_Ratio'] = df['volume'] / df['Volume_MA']

            # 计算 ATR 和波动率
            df['ATR'] = self.calculate_atr(df, self.params['atr_period'])
            df['Volatility'] = df['ATR'] / df['close'] * 100

            # 动量指标 - 调整为周线
            df['ROC'] = df['close'].pct_change(periods=8) * 100  # 8 周变化率

            return df

        except Exception as e:
            self.logger.error(f"计算技术指标时出错: {str(e)}")
            raise

    def calculate_score(self, df):
        """计算股票评分"""
        try:
            score = 0
            latest = df.iloc[-1]

            # 趋势得分 (30 分)
            if latest['MA4'] > latest['MA12']:
                score += 15
            if latest['MA12'] > latest['MA24']:
                score += 15

            # RSI 得分 (20 分)
            if 30 <= latest['RSI'] <= 70:
                score += 20
            elif latest['RSI'] < 30:  # 超卖
                score += 15

            # MACD 得分 (20 分)
            if latest['MACD'] > latest['Signal']:
                score += 20

            # 成交量得分 (30 分)
            if latest['Volume_Ratio'] > 1.5:
                score += 30
            elif latest['Volume_Ratio'] > 1:
                score += 15

            return score

        except Exception as e:
            self.logger.error(f"计算评分时出错: {str(e)}")
            raise

    def get_recommendation(self, score):
        """根据得分给出建议"""
        if score >= 80:
            return '强烈推荐买入(周线)'
        elif score >= 60:
            return '建议买入(周线)'
        elif score >= 40:
            return '观望(周线)'
        elif score >= 20:
            return '建议卖出(周线)'
        else:
            return '强烈建议卖出(周线)'

    def analyze_stock(self, df):
        """分析单个股票"""
        try:
            # 转换为周线数据
            weekly_df = self.get_weekly_data(df)

            # 计算技术指标
            weekly_df = self.calculate_indicators(weekly_df)

            # 评分系统
            score = self.calculate_score(weekly_df)

            # 获取最新数据
            latest = weekly_df.iloc[-1]
            prev = weekly_df.iloc[-2]

            # 生成报告
            report = {
                'score': score,
                'price': latest['close'],
                'price_change': (latest['close'] - prev['close']) / prev['close'] * 100,
                'ma_trend': 'UP' if latest['MA4'] > latest['MA12'] else 'DOWN',
                'rsi': latest['RSI'],
                'macd_signal': 'BUY' if latest['MACD'] > latest['Signal'] else 'SELL',
                'volume_status': 'HIGH' if latest['Volume_Ratio'] > 1.5 else 'NORMAL',
                'recommendation': self.get_recommendation(score),
                'weekly_data': weekly_df
            }

            return report

        except Exception as e:
            self.logger.error(f"分析股票时出错: {str(e)}")
            raise

# 数据获取
@st.cache_data(ttl=3600)
def get_stock_data(symbol):
    try:
        # 获取当前日期
        end_date = datetime.now().strftime("%Y%m%d")
        # 获取个股历史数据
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date="20150101", end_date=end_date)
        if df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
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
    current_year = datetime.now().year
    current_month = datetime.now().month

    for y in range(2015, current_year + 1):
        row = {'年份': y}
        for m in range(1, 13):
            if y == current_year and m > current_month:
                break

            sub = df[(df['year'] == y) & (df['month'] == m)]
            if len(sub) >= 3:  # 至少 3 个交易日才计算
                try:
                    chg = (sub.iloc[-1]['close'] / sub.iloc[0]['close'] - 1) * 100
                    row[f'{m}月'] = round(chg, 2)
                except:
                    row[f'{m}月'] = None
            else:
                row[f'{m}月'] = None
        monthly.append(row)
    return pd.DataFrame(monthly).set_index('年份')

# 计算近 6 个月高低点
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

def plot_technical_indicators(weekly_df):
    """绘制技术指标图表"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # 价格和均线
    axes[0].plot(weekly_df['date'], weekly_df['close'], label='收盘价', color='#2c3e50', linewidth=2)
    axes[0].plot(weekly_df['date'], weekly_df['MA4'], label='4 周均线', color='#3498db', linestyle='--')
    axes[0].plot(weekly_df['date'], weekly_df['MA12'], label='12 周均线', color='#e74c3c', linestyle='--')
    axes[0].plot(weekly_df['date'], weekly_df['MA24'], label='24 周均线', color='#2ecc71', linestyle='--')
    axes[0].set_title('价格与移动平均线', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # MACD
    axes[1].plot(weekly_df['date'], weekly_df['MACD'], label='MACD', color='#3498db', linewidth=1.5)
    axes[1].plot(weekly_df['date'], weekly_df['Signal'], label='信号线', color='#e74c3c', linewidth=1.5)
    axes[1].bar(weekly_df['date'], weekly_df['MACD_hist'], 
               label='MACD 柱', 
               color=np.where(weekly_df['MACD_hist'] >= 0, '#e74c3c', '#2ecc71'),
               width=5)
    axes[1].axhline(0, color='#7f8c8d', linestyle='--')
    axes[1].set_title('MACD 指标', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # RSI
    axes[2].plot(weekly_df['date'], weekly_df['RSI'], label='RSI', color='#9b59b6', linewidth=1.5)
    axes[2].axhline(70, color='#e74c3c', linestyle='--')
    axes[2].axhline(30, color='#2ecc71', linestyle='--')
    axes[2].fill_between(weekly_df['date'], 30, 70, color='#f2e6ff', alpha=0.3)
    axes[2].set_title('RSI 指标', fontsize=12, fontweight='bold')
    axes[2].legend(loc='upper left')
    axes[2].grid(True, linestyle='--', alpha=0.7)
    axes[2].set_ylim(0, 100)
    
    # 成交量
    axes[3].bar(weekly_df['date'], weekly_df['volume'], 
               label='成交量', 
               color=np.where(weekly_df['close'] >= weekly_df['close'].shift(1), '#e74c3c', '#2ecc71'),
               width=5)
    axes[3].plot(weekly_df['date'], weekly_df['Volume_MA'], label='成交量均线', color='#f39c12', linewidth=1.5)
    axes[3].set_title('成交量', fontsize=12, fontweight='bold')
    axes[3].legend(loc='upper left')
    axes[3].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def app():
    # 页面样式设置 - 现代化设计
    st.markdown("""
    <style>
    /* 整体页面样式 */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        padding: 1.5rem 2rem;
        background-color: #f8f9fa;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    
    /* 主标题样式 */
    .stMarkdown h1 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
        font-weight: 700;
    }
    
    /* 副标题样式 */
    .stMarkdown h2 {
        color: #2c3e50;
        border-left: 4px solid #3498db;
        padding-left: 0.75rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* 副标题样式 */
    .stMarkdown h3 {
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* 指标卡样式 */
    div[data-testid="metric-container"] {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        height: 120px;
        background-color: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transform: translateY(-3px);
    }
    
    /* 指标标题 */
    label[data-testid="stMetricLabel"] p {
        font-size: 14px !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
        color: #7f8c8d;
    }
    
    /* 主数值 */
    div[data-testid="stMetricValue"] {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
    }
    
    /* Delta 值 */
    div[data-testid="stMetricDelta"] {
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    
    /* 输入框样式 */
    .stTextInput input {
        border-radius: 8px;
        border: 1px solid #dfe1e5;
        padding: 0.75rem;
    }
    
    /* 表格样式 */
    .stDataFrame {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* 分隔线样式 */
    .stMarkdown hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background-color: #e0e0e0;
    }
    
    /* 标签样式 */
    .st-bd {
        padding: 0.5rem 1rem;
    }
    
    /* 按钮样式 */
    .stButton button {
        border-radius: 8px;
        background-color: #3498db;
        color: white;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border: none;
    }
    
    .stButton button:hover {
        background-color: #2980b9;
        color: white;
    }
    
    /* 加载动画 */
    .stSpinner div {
        border-color: #3498db transparent transparent transparent;
    }
    </style>
    """, unsafe_allow_html=True)

    # 页面标题
    st.title("📈 A 股个股周线技术分析评分及月度历史数据统计")
    st.markdown("""
    <div style="color: #7f8c8d; margin-bottom: 1.5rem;">
    基于周线技术指标分析和历史月度数据统计，提供股票投资参考建议
    </div>
    """, unsafe_allow_html=True)

    # 股票代码输入
    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            stock_code = st.text_input(
                "输入 A 股股票代码",
                value="600519",
                placeholder="如: 600519 或 000001",
                help="输入 6 位数字的 A 股股票代码，SH/SZ 后缀可选"
            ).strip()
        
        # 清理股票代码输入
        stock_code = ''.join(filter(str.isdigit, stock_code))[:6]
        if not stock_code:
            st.warning("请输入有效的股票代码")
            st.stop()

    # 获取数据
    with st.spinner(f"正在获取 {stock_code} 的历史数据..."):
        df = get_stock_data(stock_code)
        if df.empty:
            st.error("未能获取该股票数据，请检查代码是否正确或稍后再试")
            st.stop()

    stock_name = get_stock_name(stock_code)
    monthly_df = build_monthly_table(df)
    prob_df = calculate_monthly_probability(monthly_df)

    # 计算各类指标
    latest_close = df.iloc[-1]['close']
    hist_high = df['close'].max()
    recent_high, recent_low = get_recent_high_low(df)
    valid_months = monthly_df.notna().sum().sum()

    # 计算 6 个月区间指标
    if recent_high and recent_low:
        range_pct = f"{(recent_high - recent_low)/recent_low*100:.1f}%"
        from_high = f"{(1 - latest_close/recent_high)*100:.1f}%"
        from_low = f"{(latest_close/recent_low - 1)*100:.1f}%"
        position_ratio = f"{(latest_close - recent_low)/(recent_high - recent_low)*100:.1f}%"
    else:
        range_pct = from_high = from_low = position_ratio = "N/A"

    # 显示股票信息
    st.markdown(f"""
    ### 🏷️ {stock_name}({stock_code})
    <div style="color: #7f8c8d; margin-bottom: 1.5rem;">
    ▸ **数据范围**: 2015 年 1 月 - {datetime.now().strftime('%Y 年%m 月')} &nbsp; | &nbsp;
    ▸ **最后更新**: {df['date'].max().strftime('%Y-%m-%d')} &nbsp; | &nbsp;
    ▸ **数据完整性**: {valid_months}/{(datetime.now().year-2015+1)*12} 个月度数据
    </div>
    """, unsafe_allow_html=True)

    # 关键指标卡片
    with st.container():
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
                delta_color="inverse",
                help="历史最高收盘价及当前差距"
            )
        with col3:
            st.metric(
                "6 个月区间",
                f"{recent_low:.2f}-{recent_high:.2f}",
                delta=f"振幅:{range_pct}",
                help="近 6 个月最低价-最高价及波动幅度"
            )
        with col4:
            st.metric(
                "区间位置",
                position_ratio,
                delta=f"↑{from_low} ↓{from_high}",
                help="当前价在 6 个月区间中的位置(0%-100%)"
            )

    # 技术分析部分
    st.markdown("---")
    st.subheader("📊 技术分析 (周线)")
    st.markdown("基于周线数据的技术指标分析，包括均线系统、MACD、RSI等指标")

    analyzer = StockAnalyzer()
    with st.spinner("正在进行技术分析..."):
        try:
            tech_report = analyzer.analyze_stock(df)

            # 技术指标卡片
            with st.container():
                cols = st.columns(4)
                with cols[0]:
                    st.metric(
                        "技术评分",
                        f"{tech_report['score']}/100",
                        delta=tech_report['recommendation'],
                        delta_color="normal",
                        help="技术分析综合评分"
                    )
                with cols[1]:
                    st.metric(
                        "趋势方向",
                        tech_report['ma_trend'],
                        delta=f"RSI: {tech_report['rsi']:.1f}",
                        help="均线趋势和 RSI 指标"
                    )
                with cols[2]:
                    st.metric(
                        "MACD 信号",
                        tech_report['macd_signal'],
                        delta=f"周涨跌: {tech_report['price_change']:.1f}%",
                        help="MACD 指标信号"
                    )
                with cols[3]:
                    st.metric(
                        "成交量状态",
                        tech_report['volume_status'],
                        delta=f"量比: {tech_report['weekly_data'].iloc[-1]['Volume_Ratio']:.1f}",
                        help="成交量状态"
                    )

            # 绘制技术指标图表
            st.markdown("#### 技术指标图表")
            fig = plot_technical_indicators(tech_report['weekly_data'])
            st.pyplot(fig)

            # 显示技术指标数据
            st.markdown("#### 最近20周技术指标数据")
            st.dataframe(
                tech_report['weekly_data'].tail(20).style.format({
                    'close': '{:.2f}',
                    'MA4': '{:.2f}',
                    'MA12': '{:.2f}',
                    'MA24': '{:.2f}',
                    'RSI': '{:.1f}',
                    'MACD': '{:.3f}',
                    'Signal': '{:.3f}',
                    'Volume_Ratio': '{:.1f}',
                    'Volatility': '{:.1f}%',
                    'ROC': '{:.1f}%'
                }).applymap(
                    lambda x: f"color: {dual_colors[1]}" if isinstance(x, (int, float)) and x >= 0 
                            else (f"color: {dual_colors[0]}" if isinstance(x, (int, float)) and x < 0 else None),
                subset=['ROC', 'price_change']
                ),
                height=600,
                use_container_width=True
            )

        except Exception as e:
            st.error(f"技术分析失败: {str(e)}")

    # 月度涨跌概率分析
    st.markdown("---")
    st.subheader("📊 月度涨跌概率分析")
    st.markdown("基于历史数据的月度涨跌概率统计，帮助识别季节性规律")

    if not prob_df.empty:
        # 找出最佳和最差月份
        best_month = prob_df['上涨概率(%)'].idxmax()
        worst_month = prob_df['上涨概率(%)'].idxmin()

        # 概率表格样式
        prob_styled = prob_df.style.format({
            '上涨概率(%)': '{:.1f}%',
            '下跌概率(%)': '{:.1f}%',
            '平均涨跌幅(%)': '{:.2f}%',
            '最大涨幅(%)': '{:.2f}%',
            '最大跌幅(%)': '{:.2f}%'
        }).background_gradient(
            cmap='RdYlGn',
            subset=['上涨概率(%)']
        ).background_gradient(
            cmap='RdYlGn',
            subset=['平均涨跌幅(%)'],
            vmin=-10, vmax=10
        )

        st.dataframe(prob_styled, height=600, use_container_width=True)

        # 分析结论
        st.markdown("### 📌 分析结论")
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #3498db;">
        - 🏆 **表现最佳月份**: **{best_month}**  
        上涨概率 {prob_df.loc[best_month, '上涨概率(%)']}%，平均涨幅 {prob_df.loc[best_month, '平均涨跌幅(%)']}%  
        最大涨幅 {prob_df.loc[best_month, '最大涨幅(%)']}%，最大跌幅 {prob_df.loc[best_month, '最大跌幅(%)']}%

        - 🚨 **表现最差月份**: **{worst_month}**  
        上涨概率 {prob_df.loc[worst_month, '上涨概率(%)']}%，平均涨幅 {prob_df.loc[worst_month, '平均涨跌幅(%)']}%  
        最大涨幅 {prob_df.loc[worst_month, '最大涨幅(%)']}%，最大跌幅 {prob_df.loc[worst_month, '最大跌幅(%)']}%

        - 📈 **整体统计**:  
        平均上涨概率: {prob_df['上涨概率(%)'].mean():.1f}%  
        平均月度涨幅: {prob_df['平均涨跌幅(%)'].mean():.2f}%  
        数据覆盖: {valid_months} 个月度数据
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("没有足够的月度数据进行分析")

    # 原始月度数据
    st.markdown("---")
    st.subheader("📅 详细月度数据")
    st.markdown("历史各月份涨跌幅数据（百分比）")

    styled_monthly = monthly_df.style.format('{:.1f}%', na_rep="-").applymap(
        lambda val: f'background-color: {dual_colors[1] if val is not None and val >= 0 else dual_colors[0]}; color: white'
    )
    st.dataframe(
        styled_monthly,
        height=600,
        use_container_width=True
    )

    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; margin-top: 2rem;">
    <small>
    数据来源: AKShare | 分析结果仅供参考，不构成投资建议 | 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </small>
    </div>
    """.format(datetime=datetime), unsafe_allow_html=True)

if __name__ == "__main__":
    app()
