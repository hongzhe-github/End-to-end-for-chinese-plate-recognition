import sys
from myui import *
from chinesePlateRecognizer import *


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


def process_image(img_src_path, unet, cnn) -> (Image, Image, str):
    """
    处理图像函数

    参数:
        img_src_path (str): 图像文件路径
        unet: 使用的unet模型
        cnn: 使用的cnn模型
    返回:
        (
        labeled_image：用于表示绘制了矩形框后的图像。该变量存储了包含矩形框标注的图像数据。
        resized_image：用于表示经过调整大小后的图像。该变量存储了调整大小后的图像数据。
        predicted_text：用于表示预测结果的文本。该变量存储了模型对于某种任务（例如图像分类或文本识别）的预测结果文本。
        )
    """

    img_src = cv2.imdecode(np.fromfile(img_src_path, dtype=np.uint8), -1)  # 从中文路径读取时用
    h, w = img_src.shape[0], img_src.shape[1]

    # 情况一: 满足该条件说明可能整个图片就是一张车牌,无需定位,直接识别即可
    if h * w <= 240 * 80 and 2 <= w / h <= 5:
        lic = cv2.resize(img_src, dsize=(240, 80), interpolation=cv2.INTER_AREA)[:, :, :3]  # 直接resize为(240,80)
        img_src_copy, Lic_img = img_src, [lic]

    # 情况二: 需通过unet对img_src原图预测,得到img_mask,实现车牌定位,然后进行识别
    else:
        img_src, img_mask = unet_predict(unet, img_src_path)
        # 利用core.py中的locate_and_correct函数进行车牌定位和矫正
        img_src_copy, Lic_img = locate_and_correct(img_src, img_mask)

    Lic_pred = cnn_predict(cnn, Lic_img)  # 利用cnn进行车牌的识别预测,Lic_pred中存的是元祖(车牌图片,识别结果)

    if Lic_pred is None: return None
    # 识别成功

    labeled_image = Image.fromarray(img_src_copy[:, :, ::-1])
    resized_image = Image.fromarray(Lic_pred[0][0][:, :, ::-1])
    predicted_text = Lic_pred[0][1]

    return labeled_image, resized_image, predicted_text


def close():
    keras.backend.clear_session()
    sys.exit(0)


a = MyLpr(*cnn_init())
a.protocol("WM_DELETE_WINDOW", close)
a.mainloop()
