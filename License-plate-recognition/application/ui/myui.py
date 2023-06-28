import tkinter as tk
from tkinter import Label, Canvas, Button, StringVar, Entry
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image


class MyUI(tk.Tk):

    def __init__(self, unet, cnn):
        super().__init__()

        [self.button_clear_all,
         self.img_src_path,
         self.button_recognize_plate,
         self.button_select_file,
         self.can_pred3,
         self.can_lic3,
         self.can_pred2,
         self.can_lic2,
         self.can_pred1,
         self.can_lic1,
         self.can_src] = [0] * 11

        self.unet, self.cnn = unet, cnn

        self.setupUI(1000, 600)  # 窗口宽设定1000 窗口高设定600

    def setupUI(self, window_width, window_height):
        self.img_src_path = None
        self.geometry("%dx%d+%d+%d" % (window_width, window_height, 200, 50))  # 界面启动时的初始位置
        self.title("车牌定位，矫正和识别软件")
        self.img_src_path = None

        # Label(self, text='原图:', font=('微软雅黑', 13)).place(x=0, y=0)
        Label(self, text='车牌区域1:', font=('微软雅黑', 13)).place(x=615, y=0)
        Label(self, text='识别结果1:', font=('微软雅黑', 13)).place(x=615, y=85)
        Label(self, text='车牌区域2:', font=('微软雅黑', 13)).place(x=615, y=180)
        Label(self, text='识别结果2:', font=('微软雅黑', 13)).place(x=615, y=265)
        Label(self, text='车牌区域3:', font=('微软雅黑', 13)).place(x=615, y=360)
        Label(self, text='识别结果3:', font=('微软雅黑', 13)).place(x=615, y=445)

        self.can_src = Canvas(self, width=512, height=512, bg='white', relief='solid', borderwidth=1)  # 原图画布
        self.can_src.place(x=50, y=0)
        self.can_lic1 = Canvas(self, width=245, height=85, bg='white', relief='solid', borderwidth=1)  # 车牌区域1画布
        self.can_lic1.place(x=710, y=0)
        self.can_pred1 = Canvas(self, width=245, height=65, bg='white', relief='solid', borderwidth=1)  # 车牌识别1画布
        self.can_pred1.place(x=710, y=90)
        self.can_lic2 = Canvas(self, width=245, height=85, bg='white', relief='solid', borderwidth=1)  # 车牌区域2画布
        self.can_lic2.place(x=710, y=175)
        self.can_pred2 = Canvas(self, width=245, height=65, bg='white', relief='solid', borderwidth=1)  # 车牌识别2画布
        self.can_pred2.place(x=710, y=265)
        self.can_lic3 = Canvas(self, width=245, height=85, bg='white', relief='solid', borderwidth=1)  # 车牌区域3画布
        self.can_lic3.place(x=710, y=350)
        self.can_pred3 = Canvas(self, width=245, height=65, bg='white', relief='solid', borderwidth=1)  # 车牌识别3画布
        self.can_pred3.place(x=710, y=440)

        self.button_select_file = Button(self, text='选择文件', width=10, height=1,
                                         command=self.load_show_img)  # 选择文件按钮
        self.button_select_file.place(x=680, y=window_height - 30)
        self.button_recognize_plate = Button(self, text='识别车牌', width=10, height=1, command=self.display)  # 识别车牌按钮
        self.button_recognize_plate.place(x=780, y=window_height - 30)
        self.button_clear_all = Button(self, text='清空所有', width=10, height=1, command=self.clear)  # 清空所有按钮
        self.button_clear_all.place(x=880, y=window_height - 30)

    # 加载文件
    def load_show_img(self):
        self.clear()
        sv = StringVar()
        sv.set(askopenfilename())
        self.img_src_path = Entry(self, state='readonly', text=sv).get()  # 获取到所打开的图片

        img_open = Image.open(self.img_src_path)

        if img_open.size[0] * img_open.size[1] > 240 * 80:
            img_open = img_open.resize((512, 512), Image.LANCZOS)

        self.img_Tk = ImageTk.PhotoImage(img_open)
        self.can_src.create_image(258, 258, image=self.img_Tk, anchor='center')

    # 显示
    def display(self):
        raise NotImplementedError

    # 清除
    def clear(self):
        self.can_src.delete('all')
        self.can_lic1.delete('all')
        self.can_lic2.delete('all')
        self.can_lic3.delete('all')
        self.can_pred1.delete('all')
        self.can_pred2.delete('all')
        self.can_pred3.delete('all')
        self.img_src_path = None


