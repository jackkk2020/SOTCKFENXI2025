import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLineEdit, QPushButton, QTextBrowser,
                           QLabel, QTextEdit, QMessageBox, QProgressBar, 
                           QFrame, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor
import markdown2
from stock_analyzer import StockAnalyzer

class ModernFrame(QFrame):
    """现代化的面板组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setStyleSheet("""
            ModernFrame {
                background-color: #ffffff;
                border-radius: 10px;
                border: 1px solid #e0e0e0;
            }
        """)

class ModernButton(QPushButton):
    """现代化的按钮组件"""
    def __init__(self, text, parent=None, primary=True):
        super().__init__(text, parent)
        self.setMinimumHeight(40)
        if primary:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #1a73e8;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 8px 16px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #1557b0;
                }
                QPushButton:pressed {
                    background-color: #0d47a1;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #f8f9fa;
                    color: #1a73e8;
                    border: 1px solid #dadce0;
                    border-radius: 5px;
                    padding: 8px 16px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #f1f3f4;
                    border-color: #d2e3fc;
                }
                QPushButton:pressed {
                    background-color: #e8eaed;
                }
                QPushButton:disabled {
                    color: #5f6368;
                    border-color: #e0e0e0;
                }
            """)

class ModernLineEdit(QLineEdit):
    """现代化的输入框组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(40)
        self.setStyleSheet("""
            QLineEdit {
                border: 2px solid #e0e0e0;
                border-radius: 5px;
                padding: 8px 12px;
                background-color: white;
                selection-background-color: #cce0ff;
            }
            QLineEdit:focus {
                border-color: #1a73e8;
            }
            QLineEdit:hover {
                border-color: #999999;
            }
        """)

class ModernTextEdit(QTextEdit):
    """现代化的多行文本输入框组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QTextEdit {
                border: 2px solid #e0e0e0;
                border-radius: 5px;
                padding: 8px;
                background-color: white;
                selection-background-color: #cce0ff;
            }
            QTextEdit:focus {
                border-color: #1a73e8;
            }
            QTextEdit:hover {
                border-color: #999999;
            }
        """)

class ModernProgressBar(QProgressBar):
    """现代化的进度条组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 3px;
                background-color: #f0f0f0;
                height: 6px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #1a73e8;
                border-radius: 3px;
            }
        """)
        self.setTextVisible(False)

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

class ModernStockAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = StockAnalyzer()
        self.init_ui()
        self.adjust_size_and_position()

    def adjust_size_and_position(self):
        """调整窗口大小和位置以适应不同分辨率"""
        screen = QApplication.primaryScreen()
        if screen:
            geometry = screen.availableGeometry()
            # 设置窗口大小为屏幕的75%
            width = int(geometry.width() * 0.75)
            height = int(geometry.height() * 0.75)
            self.resize(width, height)
            
            # 居中显示
            center = geometry.center()
            frame = self.frameGeometry()
            frame.moveCenter(center)
            self.move(frame.topLeft())

    def init_ui(self):
        self.setWindowTitle('现代股票分析系统')
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
        """)

        # 创建中央部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 创建标题
        title_label = QLabel('股票分析系统')
        title_label.setStyleSheet("""
            QLabel {
                color: #202124;
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 20px;
            }
        """)
        main_layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignTop)

        # 创建输入区域容器
        input_container = ModernFrame()
        input_layout = QHBoxLayout(input_container)
        input_layout.setSpacing(20)
        input_layout.setContentsMargins(20, 20, 20, 20)

        # 单只股票分析部分
        single_stock_frame = self.create_single_stock_section()
        input_layout.addWidget(single_stock_frame)

        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setStyleSheet("background-color: #e0e0e0;")
        input_layout.addWidget(separator)

        # 批量分析部分
        batch_stock_frame = self.create_batch_stock_section()
        input_layout.addWidget(batch_stock_frame)

        main_layout.addWidget(input_container)

        # 进度条
        self.progress_bar = ModernProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 结果显示区域
        result_frame = ModernFrame()
        result_layout = QVBoxLayout(result_frame)
        result_layout.setContentsMargins(15, 15, 15, 15)

        result_label = QLabel('分析结果')
        result_label.setStyleSheet("""
            QLabel {
                color: #202124;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        result_layout.addWidget(result_label)

        self.result_browser = QTextBrowser()
        self.result_browser.setOpenExternalLinks(True)
        self.result_browser.setStyleSheet("""
            QTextBrowser {
                border: none;
                background-color: white;
                font-size: 14px;
                line-height: 1.5;
            }
        """)
        result_layout.addWidget(self.result_browser)

        main_layout.addWidget(result_frame)

    def create_single_stock_section(self):
        """创建单只股票分析部分"""
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setSpacing(15)

        label = QLabel('单只股票分析')
        label.setStyleSheet("""
            QLabel {
                color: #202124;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        layout.addWidget(label)

        self.single_stock_input = ModernLineEdit()
        self.single_stock_input.setPlaceholderText('输入股票代码（如：600000）')
        layout.addWidget(self.single_stock_input)

        self.analyze_btn = ModernButton('分析')
        self.analyze_btn.clicked.connect(self.analyze_single_stock)
        layout.addWidget(self.analyze_btn)

        layout.addStretch()
        return frame

    def create_batch_stock_section(self):
        """创建批量分析部分"""
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setSpacing(15)

        label = QLabel('批量股票分析')
        label.setStyleSheet("""
            QLabel {
                color: #202124;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        layout.addWidget(label)

        self.batch_stock_input = ModernTextEdit()
        self.batch_stock_input.setPlaceholderText('输入多个股票代码，每行一个')
        self.batch_stock_input.setMinimumHeight(100)
        layout.addWidget(self.batch_stock_input)

        self.batch_analyze_btn = ModernButton('批量分析')
        self.batch_analyze_btn.clicked.connect(self.analyze_multiple_stocks)
        layout.addWidget(self.batch_analyze_btn)

        layout.addStretch()
        return frame

    def format_report(self, report, is_single=True):
        """将分析报告格式化为现代化的Markdown格式"""
        md = f"""# 股票分析报告 - {report['stock_code']}

## 基本信息
- **分析日期：** {report['analysis_date']}
- **当前价格：** ¥{report['price']:.2f}
- **价格变动：** {report['price_change']:.2f}%

## 技术指标
- **均线趋势：** {report['ma_trend']}
- **RSI指标：** {report['rsi']:.2f}
- **MACD信号：** {report['macd_signal']}
- **成交量状态：** {report['volume_status']}

## 评分与建议
- **综合评分：** {report['score']}分
- **投资建议：** {report['recommendation']}

## AI分析
{report['ai_analysis']}

---
"""
        return md

    def analyze_single_stock(self):
        """分析单只股票"""
        stock_code = self.single_stock_input.text().strip()
        if not stock_code:
            self.show_warning('请输入股票代码')
            return

        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.worker = AnalysisWorker(self.analyzer, stock_code)
        self.worker.finished.connect(self.handle_single_analysis_result)
        self.worker.error.connect(self.handle_analysis_error)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.start()

    def analyze_multiple_stocks(self):
        """批量分析股票"""
        text = self.batch_stock_input.toPlainText().strip()
        if not text:
            self.show_warning('请输入股票代码')
            return

        stock_list = [code.strip() for code in text.split('\n') if code.strip()]
        
        self.batch_analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.batch_worker = BatchAnalysisWorker(self.analyzer, stock_list)
        self.batch_worker.finished.connect(self.handle_batch_analysis_result)
        self.batch_worker.error.connect(self.handle_analysis_error)
        self.batch_worker.progress.connect(self.progress_bar.setValue)
        self.batch_worker.start()

    def handle_single_analysis_result(self, report):
        """处理单只股票分析结果"""
        markdown_text = self.format_report(report)
        html_content = markdown2.markdown(markdown_text, extras=['tables', 'fenced-code-blocks'])
        self.result_browser.setHtml(html_content)
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def handle_batch_analysis_result(self, recommendations):
        """处理批量分析结果"""
        markdown_text = "# 批量股票分析报告\n\n"
        for rec in recommendations:
            markdown_text += self.format_report(rec, False)
            
        html_content = markdown2.markdown(markdown_text, extras=['tables', 'fenced-code-blocks'])
        self.result_browser.setHtml(html_content)
        self.batch_analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
    def handle_analysis_error(self, error_message):
        """处理分析错误"""
        self.show_error(f'分析过程中出现错误：{error_message}')
        self.analyze_btn.setEnabled(True)
        self.batch_analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def show_warning(self, message):
        """显示警告对话框"""
        warning = QMessageBox(self)
        warning.setIcon(QMessageBox.Icon.Warning)
        warning.setWindowTitle('警告')
        warning.setText(message)
        warning.setStandardButtons(QMessageBox.StandardButton.Ok)
        warning.setStyleSheet("""
            QMessageBox {
                background-color: white;
            }
            QMessageBox QLabel {
                color: #202124;
                min-width: 200px;
            }
            QPushButton {
                background-color: #1a73e8;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #1557b0;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
        """)
        warning.exec()

    def show_error(self, message):
        """显示错误对话框"""
        error = QMessageBox(self)
        error.setIcon(QMessageBox.Icon.Critical)
        error.setWindowTitle('错误')
        error.setText(message)
        error.setStandardButtons(QMessageBox.StandardButton.Ok)
        error.setStyleSheet("""
            QMessageBox {
                background-color: white;
            }
            QMessageBox QLabel {
                color: #202124;
                min-width: 200px;
            }
            QPushButton {
                background-color: #1a73e8;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #1557b0;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
        """)
        error.exec()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 创建并显示主窗口
    window = ModernStockAnalyzerGUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
