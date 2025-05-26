import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLineEdit, QPushButton, QTextBrowser,
                           QLabel, QTextEdit, QMessageBox, QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
import markdown2
from stock_analyzer import StockAnalyzer  # 导入股票分析器

class AnalysisWorker(QThread):
    """后台工作线程，用于执行分析任务"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, analyzer, stock_code):
        super().__init__()
        self.analyzer = analyzer
        self.stock_code = stock_code

    def run(self):
        try:
            report = self.analyzer.analyze_stock(self.stock_code)
            self.finished.emit(report)
        except Exception as e:
            self.error.emit(str(e))

class BatchAnalysisWorker(QThread):
    """后台工作线程，用于执行批量分析任务"""
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, analyzer, stock_list):
        super().__init__()
        self.analyzer = analyzer
        self.stock_list = stock_list

    def run(self):
        try:
            results = []
            total = len(self.stock_list)
            for i, stock_code in enumerate(self.stock_list):
                report = self.analyzer.analyze_stock(stock_code)
                results.append(report)
                self.progress.emit(int((i + 1) / total * 100))
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

class StockAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = StockAnalyzer()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('股票分析系统')
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 创建输入区域
        input_layout = QHBoxLayout()
        
        # 单只股票分析部分
        single_stock_layout = QVBoxLayout()
        single_label = QLabel('单只股票分析:')
        single_label.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        self.single_stock_input = QLineEdit()
        self.single_stock_input.setPlaceholderText('输入股票代码（如：600000）')
        self.single_stock_input.setFont(QFont('Arial', 10))
        self.analyze_btn = QPushButton('分析')
        self.analyze_btn.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        self.analyze_btn.clicked.connect(self.analyze_single_stock)
        single_stock_layout.addWidget(single_label)
        single_stock_layout.addWidget(self.single_stock_input)
        single_stock_layout.addWidget(self.analyze_btn)
        
        # 批量分析部分
        batch_stock_layout = QVBoxLayout()
        batch_label = QLabel('批量股票分析:')
        batch_label.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        self.batch_stock_input = QTextEdit()
        self.batch_stock_input.setPlaceholderText('输入多个股票代码，每行一个')
        self.batch_stock_input.setFont(QFont('Arial', 10))
        self.batch_stock_input.setMaximumHeight(100)
        self.batch_analyze_btn = QPushButton('批量分析')
        self.batch_analyze_btn.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        self.batch_analyze_btn.clicked.connect(self.analyze_multiple_stocks)
        batch_stock_layout.addWidget(batch_label)
        batch_stock_layout.addWidget(self.batch_stock_input)
        batch_stock_layout.addWidget(self.batch_analyze_btn)
        
        input_layout.addLayout(single_stock_layout)
        input_layout.addLayout(batch_stock_layout)
        
        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # 添加结果显示区域
        self.result_browser = QTextBrowser()
        self.result_browser.setOpenExternalLinks(True)
        self.result_browser.setFont(QFont('Arial', 10))
        
        layout.addLayout(input_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.result_browser)
        
    def format_report(self, report, is_single=True):
        """将分析报告格式化为Markdown格式"""
        md = f"""# 股票分析报告 - {report['stock_code']}

## 基本信息
- 分析日期：{report['analysis_date']}
- 当前价格：{report['price']:.2f}
- 价格变动：{report['price_change']:.2f}%

## 技术指标
- 均线趋势：{report['ma_trend']}
- RSI指标：{report['rsi']:.2f}
- MACD信号：{report['macd_signal']}
- 成交量状态：{report['volume_status']}

## 评分与建议
- 综合评分：{report['score']}分
- 投资建议：{report['recommendation']}

## AI分析
{report['ai_analysis']}

---
"""
        return md
        
    def analyze_single_stock(self):
        """分析单只股票"""
        stock_code = self.single_stock_input.text().strip()
        if not stock_code:
            QMessageBox.warning(self, '警告', '请输入股票代码')
            return
            
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 创建工作线程
        self.worker = AnalysisWorker(self.analyzer, stock_code)
        self.worker.finished.connect(self.handle_single_analysis_result)
        self.worker.error.connect(self.handle_analysis_error)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.start()
        
    def handle_single_analysis_result(self, report):
        """处理单只股票分析结果"""
        markdown_text = self.format_report(report)
        html_content = markdown2.markdown(markdown_text)
        self.result_browser.setHtml(html_content)
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
            
    def analyze_multiple_stocks(self):
        """批量分析股票"""
        text = self.batch_stock_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, '警告', '请输入股票代码')
            return
            
        stock_list = [code.strip() for code in text.split('\n') if code.strip()]
        
        self.batch_analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 创建工作线程
        self.batch_worker = BatchAnalysisWorker(self.analyzer, stock_list)
        self.batch_worker.finished.connect(self.handle_batch_analysis_result)
        self.batch_worker.error.connect(self.handle_analysis_error)
        self.batch_worker.progress.connect(self.progress_bar.setValue)
        self.batch_worker.start()
        
    def handle_batch_analysis_result(self, recommendations):
        """处理批量分析结果"""
        # 生成markdown格式的报告
        markdown_text = "# 批量股票分析报告\n\n"
        for rec in recommendations:
            markdown_text += self.format_report(rec, False)
            
        html_content = markdown2.markdown(markdown_text)
        self.result_browser.setHtml(html_content)
        self.batch_analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
    def handle_analysis_error(self, error_message):
        """处理分析错误"""
        QMessageBox.critical(self, '错误', f'分析过程中出现错误：{error_message}')
        self.analyze_btn.setEnabled(True)
        self.batch_analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    # 创建并显示主窗口
    window = StockAnalyzerGUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
