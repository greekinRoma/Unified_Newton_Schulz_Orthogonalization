import sys
import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTreeWidget, QTreeWidgetItem,
                             QSplitter, QTextEdit, QFileDialog, QLabel, QHeaderView)
from PyQt5.QtCore import Qt

class WeightViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.weights = None
        
    def initUI(self):
        self.setWindowTitle('PyTorch权重查看器')
        self.setGeometry(100, 100, 1200, 800)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧面板 - 权重树形结构
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(['权重名称', '形状', '数据类型'])
        self.tree_widget.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree_widget.itemClicked.connect(self.on_item_clicked)
        
        # 右侧面板 - 详细信息
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        
        self.stats_label = QLabel('选择权重项查看详细信息')
        
        right_layout.addWidget(QLabel('详细信息:'))
        right_layout.addWidget(self.detail_text)
        right_layout.addWidget(QLabel('统计信息:'))
        right_layout.addWidget(self.stats_label)
        
        splitter.addWidget(self.tree_widget)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        
        main_layout.addWidget(splitter)
        
        # 底部按钮
        button_layout = QHBoxLayout()
        load_btn = QPushButton('加载权重文件')
        load_btn.clicked.connect(self.load_weights)
        
        close_btn = QPushButton('关闭')
        close_btn.clicked.connect(self.close)
        
        button_layout.addWidget(load_btn)
        button_layout.addWidget(close_btn)
        
        main_layout.addLayout(button_layout)
        
    def load_weights(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, '打开PyTorch权重文件', '', 'PyTorch Files (*.pt *.pth)')
        
        if file_path:
            try:
                # 加载权重
                self.weights = torch.load(file_path, map_location='cpu')
                self.populate_tree()
            except Exception as e:
                self.detail_text.setText(f"加载文件时出错: {str(e)}")
    
    def populate_tree(self):
        self.tree_widget.clear()
        
        if isinstance(self.weights, dict):
            # 处理状态字典
            if 'state_dict' in self.weights:
                # 如果是包含state_dict的检查点文件
                state_dict = self.weights['state_dict']
                root_item = QTreeWidgetItem(self.tree_widget, ['state_dict', '', ''])
            else:
                # 普通状态字典
                state_dict = self.weights
                root_item = QTreeWidgetItem(self.tree_widget, ['权重', '', ''])
                
            for key, value in state_dict.items():
                if hasattr(value, 'shape') and hasattr(value, 'dtype'):
                    item = QTreeWidgetItem(root_item, [key, str(tuple(value.shape)), str(value.dtype)])
                else:
                    item = QTreeWidgetItem(root_item, [key, '非张量', str(type(value))])
                    
        elif isinstance(self.weights, list):
            # 处理列表
            root_item = QTreeWidgetItem(self.tree_widget, ['列表', f'长度: {len(self.weights)}', ''])
            for i, item in enumerate(self.weights):
                if torch.is_tensor(item):
                    sub_item = QTreeWidgetItem(root_item, [f'[{i}]', str(tuple(item.shape)), str(item.dtype)])
                else:
                    sub_item = QTreeWidgetItem(root_item, [f'[{i}]', '非张量', str(type(item))])
        
        else:
            # 其他类型
            root_item = QTreeWidgetItem(self.tree_widget, ['权重', str(type(self.weights)), ''])
            
        self.tree_widget.expandAll()
    
    def on_item_clicked(self, item, column):
        if not item.parent():  # 跳过根项目
            return
            
        key_path = []
        current = item
        while current.parent():
            key_path.insert(0, current.text(0))
            current = current.parent()
        
        try:
            # 获取选中的权重
            weight = self.weights
            for key in key_path:
                if key.startswith('[') and key.endswith(']'):  # 列表索引
                    index = int(key[1:-1])
                    weight = weight[index]
                else:  # 字典键
                    if isinstance(weight, dict):
                        weight = weight[key]
                    elif hasattr(weight, key):
                        weight = getattr(weight, key)
                    else:
                        self.detail_text.setText(f"无法访问: {key}")
                        return
            
            # 显示详细信息
            if torch.is_tensor(weight):
                self.display_tensor_info(weight, key_path[-1])
            else:
                self.detail_text.setText(f"类型: {type(weight)}\n值: {weight}")
                self.stats_label.setText("非张量数据")
                
        except Exception as e:
            self.detail_text.setText(f"获取数据时出错: {str(e)}")
    
    def display_tensor_info(self, tensor, name):
        # 显示张量详细信息
        info_text = f"名称: {name}\n"
        info_text += f"形状: {tuple(tensor.shape)}\n"
        info_text += f"数据类型: {tensor.dtype}\n"
        info_text += f"设备: {tensor.device}\n"
        info_text += f"是否需要梯度: {tensor.requires_grad}\n\n"
        
        # 显示部分数据（避免显示过大张量）
        if tensor.numel() <= 100:
            info_text += "数据:\n"
            info_text += str(tensor.numpy())
        else:
            info_text += "数据预览 (前100个元素):\n"
            info_text += str(tensor.flatten()[:100].numpy())
            info_text += f"\n\n... 和另外 {tensor.numel() - 100} 个元素"
        
        self.detail_text.setText(info_text)
        
        # 显示统计信息
        stats_text = f"最小值: {tensor.min().item():.6f}\n"
        stats_text += f"最大值: {tensor.max().item():.6f}\n"
        stats_text += f"平均值: {tensor.mean().item():.6f}\n"
        stats_text += f"标准差: {tensor.std().item():.6f}\n"
        stats_text += f"元素总数: {tensor.numel()}"
        
        self.stats_label.setText(stats_text)

def main():
    app = QApplication(sys.argv)
    viewer = WeightViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()