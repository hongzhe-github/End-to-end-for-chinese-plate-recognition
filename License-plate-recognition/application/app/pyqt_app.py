import os.path
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog

from impl.my_impl import process_image
from orc.chinesePlateRecognizer import cnn_init
from ui.ui_main import Ui_MainWindow


# 创建 MainWindow 类并继承 Ui_MainWindow
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()

        # 设置界面
        self.setupUi(self)
        self.setWindowIcon(QIcon("../ui/img/logo.png"))
        # 绑定按钮槽函数
        self.pushButton_1.clicked.connect(self.orc_button)  # 识别车牌
        self.pushButton_2.clicked.connect(self.clear)  # 清除
        self.action.triggered.connect(self.openFile)  # 打开文件

        # 添加图片到 label_1
        pixmap1 = QPixmap('../ui/img/1.jpg')  # 替换为实际的图片路径
        pixmap1 = pixmap1.scaled(self.label_1.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.label_1.setPixmap(pixmap1)

        # 添加图片到 label_2
        pixmap2 = QPixmap('../ui/img/2.png')  # 替换为实际的图片路径
        pixmap2 = pixmap2.scaled(self.label_2.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.label_2.setPixmap(pixmap2)

        # 初始化文件路径
        self.file_path = '../ui/img/1.jpg'
        # 初始化cnn
        self.unet, self.cnn = cnn_init()

    def openFile(self):
        # self指向自身，"Open File"为文件名，"./"为当前路径，最后为文件类型筛选器
        fname, ftype = QFileDialog.getOpenFileName(self, "Open File", "../ui/img",
                                                   "JPG(*.jpg);;PNG(*.png)")  # 如果添加一个内容则需要加两个分号
        if os.path.isfile(fname):
            self.file_path = fname
            pixmap1 = QPixmap(fname).scaled(self.label_1.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.label_1.setPixmap(pixmap1)

    def orc_button(self):
        labeled_image, resized_image, predicted_text = process_image(self.file_path, self.unet, self.cnn)

        if labeled_image is not None:
            # 将 labeled_image 和 resized_image 转换为 QImage
            labeled_data = labeled_image.convert("RGBA").tobytes()
            resized_data = resized_image.convert("RGBA").tobytes()
            qimage_labeled = QImage(labeled_data, labeled_image.size[0], labeled_image.size[1], QImage.Format_RGBA8888)
            qimage_resized = QImage(resized_data, resized_image.size[0], resized_image.size[1], QImage.Format_RGBA8888)

            # 将 QImage 转换为 QPixmap
            pixmap_labeled = QPixmap.fromImage(qimage_labeled)
            pixmap_resized = QPixmap.fromImage(qimage_resized)

            # 设置 QPixmap 到 label_1 和 label_2 上显示
            self.label_1.setPixmap(pixmap_labeled)
            self.label_2.setPixmap(pixmap_resized)

            # 设置预测文本到 label_3 上显示
            self.label_3.setText(predicted_text)

    def clear(self):
        self.label_1.clear()
        self.label_2.clear()
        self.label_3.clear()


if __name__ == "__main__":
    # 创建应用程序实例
    app = QApplication(sys.argv)
    # 创建 MainWindow 实例
    window = MainWindow()
    # 显示主窗口
    window.show()
    # 开始应用程序的主循环
    sys.exit(app.exec())
