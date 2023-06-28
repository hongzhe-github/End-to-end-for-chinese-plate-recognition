import sys

from PIL import ImageTk
from tensorflow import keras

from impl.my_impl import process_image
from orc.chinesePlateRecognizer import cnn_init
from ui.myui import MyUI


class MyLpr(MyUI):
    def __int__(self):
        super().__init__(*cnn_init())

    def display(self):
        # 判断是否选择了图片
        if self.img_src_path is None:
            self.can_pred1.create_text(32, 15, text='请选择图片', anchor='nw', font=('黑体', 28))
            return

        self.can_src.delete('all')  # 显示前,先清空画板
        labeled_image, resized_image, predicted_text = process_image(self.img_src_path, self.unet, self.cnn)
        self.img_Tk = ImageTk.PhotoImage(labeled_image)
        self.lic_Tk1 = ImageTk.PhotoImage(resized_image)
        self.can_src.create_image(258, 258, image=self.img_Tk, anchor='center')  # img_src_copy上绘制出了定位的车牌轮廓,将其显示在画板上
        self.can_lic1.create_image(5, 5, image=self.lic_Tk1, anchor='nw')
        self.can_pred1.create_text(35, 15, text=predicted_text, anchor='nw', font=('黑体', 28))


def close():
    keras.backend.clear_session()
    sys.exit(0)


a = MyLpr(*cnn_init())
a.protocol("WM_DELETE_WINDOW", close)
a.mainloop()
