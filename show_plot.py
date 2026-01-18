import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import torch
from torch import nn
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
class PolynomialCalculator:
    def __init__(self):
        # 初始化多项式参数
        self.weight = np.array([ 4.8378e-01, -4.9445e-01,  7.6665e-01,  9.0778e-01,  1.7709e+00,
        -2.4217e+00,  2.3691e+00, -5.2059e+00,  6.8240e+00, -3.6552e+00,
         2.6964e+01, -5.3684e-07,  3.2497e-06,  1.4080e+02])
    def polynomial(self, x,a=None,b=None,c=None,d=None,e=None,f=None):
        """计算多项式函数值"""
        weight = np.abs(self.weight.copy())
        y = 1-x**2
        return weight[0]*x*y + weight[1]*x*y**2 + weight[2]*x*y**4 + weight[3]*x*y**8 + weight[4]*x*y**16 + weight[5]*x*y**32 + weight[6]*x*y**64 + weight[7]*x*y**128 + weight[8]*x*y**256 + weight[9]*x*y**512 + weight[10]*x*y**1024 + weight[11]*x*y**2048 + weight[12]*x*y**4096 + weight[13]*x*y**8192 + x


class PolynomialApp:
    def __init__(self, root):
        self.root = root
        self.root.title("多项式函数计算器")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # 创建多项式计算器实例
        self.calculator = PolynomialCalculator()
        
        # 设置样式
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground='#333333')
        
        # 创建主框架
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # 创建控制面板
        self.create_control_panel()
        
        # 创建结果和图表区域
        self.create_result_area()
        
        # 初始化图表
        self.update_chart()
    
    def create_control_panel(self):
        """创建参数控制面板"""
        control_frame = ttk.LabelFrame(self.main_frame, text="多项式参数控制")
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        
        # 参数标签和滑动条
        params = [
            # ('a (常数项)', 'a', 0.0, 10.0, self.calculator.a),
            # ('b (x²系数)', 'b', 0.0, 10.0, self.calculator.b),
            # ('c (x⁴系数)', 'c', 0.0, 10.0, self.calculator.c),
            # ('d (x⁶系数)', 'd', 0.0, 10.0, self.calculator.d),
        ]
        
        self.sliders = {}
        self.entries = {}
        
        for i, (label_text, param, min_val, max_val, default_val) in enumerate(params):
            # 标签
            label = ttk.Label(control_frame, text=label_text)
            label.grid(row=i, column=0, padx=5, pady=5, sticky='w')
            
            # 滑动条
            slider = ttk.Scale(control_frame, from_=min_val, to=max_val, 
                              length=200, orient='horizontal',
                              command=lambda val, p=param: self.slider_changed(p, val))
            slider.set(default_val)
            slider.grid(row=i, column=1, padx=5, pady=5, sticky='ew')
            self.sliders[param] = slider
            
            # 输入框
            var = tk.DoubleVar(value=default_val)
            var.trace_add('write', lambda *args, p=param: self.entry_changed(p))
            entry = ttk.Entry(control_frame, textvariable=var, width=8)
            entry.grid(row=i, column=2, padx=5, pady=5)
            self.entries[param] = var
        
        # X值输入
        x_frame = ttk.Frame(control_frame)
        x_frame.grid(row=len(params), column=0, columnspan=3, pady=10, sticky='ew')
        
        ttk.Label(x_frame, text="输入 x 值:").pack(side='left', padx=(0, 5))
        
        self.x_var = tk.DoubleVar(value=1.0)
        x_entry = ttk.Entry(x_frame, textvariable=self.x_var, width=10)
        x_entry.pack(side='left', padx=(0, 10))
        
        calc_button = ttk.Button(x_frame, text="计算", command=self.calculate)
        calc_button.pack(side='left')
        
        # 结果显示
        self.result_var = tk.StringVar(value="结果将显示在这里")
        result_label = ttk.Label(control_frame, textvariable=self.result_var, 
                               font=('Arial', 11, 'bold'), foreground='blue')
        result_label.grid(row=len(params)+1, column=0, columnspan=3, pady=10)
    
    def create_result_area(self):
        """创建结果和图表显示区域"""
        # 结果框架
        result_frame = ttk.LabelFrame(self.main_frame, text="函数曲线")
        result_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky='nsew')
        
        # 创建图表
        self.fig, self.ax = plt.subplots(figsize=(7, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=result_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # 函数信息
        info_frame = ttk.LabelFrame(self.main_frame, text="多项式函数")
        info_frame.grid(row=1, column=0, padx=10, pady=10, sticky='ew')
        
        info_text = "多项式函数定义：\n"
        info_text += "f(x) = (a - b·x² + c·x⁴ - d·x⁶ + e·x⁷) * x\n\n"
        info_text += "其中：\n"
        info_text += "a = 常数项\n"
        info_text += "b = x²项的系数\n"
        info_text += "c = x⁴项的系数\n"
        info_text += "d = x⁶项的系数\n"
        info_text += "e = x⁷项的系数"
        
        info_label = ttk.Label(info_frame, text=info_text, justify='left')
        info_label.pack(padx=10, pady=10)
        
        # 配置网格布局权重
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=2)
        self.main_frame.rowconfigure(0, weight=2)
        self.main_frame.rowconfigure(1, weight=1)
    
    def slider_changed(self, param, value):
        """滑动条值改变时的回调函数"""
        value = float(value)
        self.entries[param].set(round(value, 3))
        self.update_parameter(param, value)
    
    def entry_changed(self, param):
        """输入框值改变时的回调函数"""
        try:
            value = float(self.entries[param].get())
            self.sliders[param].set(value)
            self.update_parameter(param, value)
        except ValueError:
            pass
    
    def update_parameter(self, param, value):
        """更新参数值并重新计算"""
        setattr(self.calculator, param, value)
        self.update_chart()
    
    def calculate(self):
        """计算给定x值的多项式结果"""
        try:
            x = float(self.x_var.get())
            result = self.calculator.polynomial(x)
            self.result_var.set(f"f({x}) = {result:.6f}")
            self.update_chart(x)
        except ValueError:
            self.result_var.set("错误：请输入有效的数字")
    
    def update_chart(self, highlight_x=None):
        """更新函数曲线图"""
        # 清除之前的图表
        self.ax.clear()
        
        # 生成x值范围
        x = np.linspace(-0., 2., 1000)
        y = []
        for xi in x:
            yi = self.calculator.polynomial(xi)
            y.append(yi)
        y = np.array(y)
        # 绘制函数曲线
        self.ax.plot(x, y, 'b-', linewidth=2, label=f'f(x)')
        
        # 高亮显示当前x值
        if highlight_x is not None:
            highlight_y = self.calculator.polynomial(highlight_x)
            self.ax.plot(highlight_x, highlight_y, 'ro', markersize=8)
            self.ax.annotate(f'({highlight_x:.2f}, {highlight_y:.2f})', 
                            xy=(highlight_x, highlight_y),
                            xytext=(highlight_x+0.2, highlight_y+0.2),
                            arrowprops=dict(facecolor='red', shrink=0.05),
                            fontsize=10)
        
        # 设置图表标题和标签
        self.ax.set_title("多项式函数曲线", fontsize=12)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.axhline(0, color='black', linewidth=0.5)
        self.ax.axvline(0, color='black', linewidth=0.5)
        self.ax.set_ylim(-1, 3)
        # 显示图例
        self.ax.legend(loc='best')
        
        # 更新图表
        self.canvas.draw()
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = PolynomialApp(root)
    app.run()