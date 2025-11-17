import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
# import pytesseract
import os
import numpy
import torch
from main import MyNet


def read_img(img_path):
    img = Image.open(img_path)
    # print(img.size)
    img = numpy.array(img)
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    return img


# 如果你安装了 Tesseract 到非默认路径，取消下面注释并修改路径
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图片文字识别工具 - OCR GUI")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")

        self.image_path = None
        self.photo = None

        self.btn_select = None
        self.btn_select_data = None
        self.btn_ocr = None
        self.img_label = None
        self.text_result = None
        self.status = None

        self.model = None

        self.create_widgets()

    def create_widgets(self):
        # === 顶部按钮区域 ===
        top_frame = tk.Frame(self.root, bg="#f0f0f0")
        top_frame.pack(pady=10)

        self.btn_select = tk.Button(top_frame, text="选择图片", command=self.select_image,
                                    font=("微软雅黑", 12), width=12, bg="#4CAF50", fg="white")
        self.btn_select.pack(side="left", padx=5)

        self.btn_select_data = tk.Button(
            top_frame, text="选择数据", command=self.select_image,
            font=("微软雅黑", 12), width=12, bg="#4CAF50", fg="white"
        )
        self.btn_select_data.pack(side="left", padx=5)

        self.btn_ocr = tk.Button(
            top_frame, text="开始识别", command=self.run_ocr,
            font=("微软雅黑", 12), width=12, bg="#2196F3",
            fg="white", state="disabled")
        self.btn_ocr.pack(side="left", padx=5)

        # === 图片显示区域 ===
        self.img_label = tk.Label(self.root, text="未选择图片", bg="white", width=60, height=20,
                                  relief="sunken", anchor="center")
        self.img_label.pack(pady=10, padx=20, fill="both", expand=False)

        # === 结果显示区域 ===
        result_frame = tk.LabelFrame(self.root, text="识别结果", font=("微软雅黑", 12), padx=10, pady=10)
        result_frame.pack(pady=10, padx=20, fill="both", expand=True)

        self.text_result = scrolledtext.ScrolledText(result_frame, font=("微软雅黑", 11), wrap=tk.WORD)
        self.text_result.pack(fill="both", expand=True)

        # === 底部状态栏 ===
        self.status = tk.Label(self.root, text="就绪", anchor="w", bg="#f0f0f0", fg="#555", font=("微软雅黑", 9))
        self.status.pack(side="bottom", fill="x")

    def select_image(self):
        self.image_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                ("所有文件", "*.*")
            ]
        )
        if not self.image_path:
            return

        # 显示图片（缩放到窗口大小）
        try:
            img = Image.open(self.image_path)
            img.thumbnail((550, 400))  # 限制最大尺寸
            self.photo = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.photo, text="")
            self.btn_ocr.config(state=tk.NORMAL)
            self.status.config(text=f"已加载: {os.path.basename(self.image_path)}")
        except Exception as e:
            messagebox.showerror("错误", f"无法打开图片：{e}")

    def run_ocr(self):
        if not self.image_path:
            return

        self.btn_ocr.config(state=tk.DISABLED)
        self.status.config(text="正在识别...")
        self.text_result.delete(1.0, tk.END)

        # try:
        #     # 读取图片
        #     img = Image.open(self.image_path)
        #
        #     # 自动下载中文语言包（如果没装）
        #     lang = 'chi_sim+eng'  # 中文简体 + 英文
        #     # pytesseract 会自动处理语言包下载（需要网络）
        #
        #     # 执行 OCR
        #     text = pytesseract.image_to_string(img, lang=lang)
        #
        #     # 显示结果
        #     self.text_result.insert(tk.END, text if text.strip() else "（未识别到文字）")
        #     self.status.config(text="识别完成！双击可复制文本")
        #
        # except Exception as e:
        #     messagebox.showerror("识别失败", f"OCR 出错：{e}")
        #     self.status.config(text="识别失败")
        #
        # finally:
        #     self.btn_ocr.config(state=tk.NORMAL)

    def load_model(self, model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device : {device}")

        model_path = "./model.pth"
        self.model = MyNet().to(device)
        # model.load_state_dict(torch.load(model_path))
        # print(f"load model successfully {model}")


# ================== 启动程序 ==================
if __name__ == "__main__":
    init_windows = tk.Tk()
    app = OCRApp(init_windows)
    init_windows.mainloop()
