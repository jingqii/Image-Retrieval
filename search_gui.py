# -*- coding: utf-8 -*-

"""
图像检索GUI应用程序

该程序提供了一个图形用户界面，用于图像检索。功能包括：
- 加载预训练的词袋模型。
- 选择一张查询图片。
- 执行基于倒排索引的快速图像检索。
- （可选）通过RANSAC进行几何验证，以重排结果。
- （可选）通过相关性反馈（标记相关/不相关图片）来优化查询。
- 可视化RANSAC的内点匹配结果。
"""

# 导入标准库
import sys  # 用于访问与Python解释器紧密相关的变量和函数，例如退出程序
import os  # 提供了与操作系统交互的功能，如文件路径操作
import cv2  # OpenCV库，用于图像处理和计算机视觉任务
import numpy as np  # NumPy库，用于高效的数值计算，特别是多维数组操作
import joblib  # 用于高效地保存和加载Python对象，特别是大型NumPy数组

# 导入scikit-learn库
from sklearn import preprocessing  # 用于数据预处理，如此处的L2归一化

# 导入PyQt5库，用于构建图形用户界面
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QScrollArea, QGridLayout, QMessageBox, QCheckBox, 
                             QGroupBox, QSpinBox)
from PyQt5.QtGui import QPixmap, QImage  # 用于处理图像和像素图
from PyQt5.QtCore import Qt, QThread, pyqtSignal  # 提供核心非GUI功能，如信号、槽、线程和枚举值

# 导入项目内部的其他模块
from vocabulary_tree import VocabularyTree  # 词汇树类，用于将图像特征量化为“视觉单词”
from Inverted_index import compute_score_with_inverted_index  # 使用倒排索引计算相似度得分的函数
from RANSAC import ransac_verification  # RANSAC几何验证函数
from Relevance_Feedback import reweight_query_vector  # 相关性反馈中用于重新加权查询向量的函数

# --- 自定义UI组件 ---

class ClickableQWidget(QWidget):
    """
    一个可以响应鼠标点击事件的自定义QWidget。
    通过继承QWidget并重写mousePressEvent，我们可以让一个普通的容器控件变得可以点击。
    """
    clicked = pyqtSignal()  # 定义一个名为'clicked'的信号，当控件被点击时会发出

    def mousePressEvent(self, event):
        """处理鼠标点击事件，并发出clicked信号。"""
        self.clicked.emit()  # 发出信号
        super().mousePressEvent(event)  # 调用父类的同名方法，以确保标准的事件处理继续进行

class MatchViewerWindow(QMainWindow):
    """
    一个用于显示RANSAC匹配结果（带有内点连线）的新窗口。
    这个窗口专门用来展示两张图片之间的特征点匹配情况。
    """
    def __init__(self, image, title="Match Visualization"):
        """
        初始化窗口。
        :param image: 要显示的OpenCV图像 (numpy array, BGR格式)。
        :param title: 窗口标题。
        """
        super().__init__()  # 调用父类的构造函数
        self.setWindowTitle(title)  # 设置窗口的标题
        
        self.scroll_area = QScrollArea()  # 创建一个滚动区域，以防图片太大无法完全显示
        self.setCentralWidget(self.scroll_area)  # 将滚动区域设置为主窗口的中心部件
        
        self.image_label = QLabel()  # 创建一个标签用于显示图片
        self.image_label.setAlignment(Qt.AlignCenter)  # 设置图片在标签内居中对齐
        self.scroll_area.setWidget(self.image_label)  # 将图片标签放入滚动区域
        self.scroll_area.setWidgetResizable(True)  # 允许滚动区域内的部件（标签）自动调整大小

        # 将OpenCV图像 (numpy array) 转换为QPixmap，以便在QLabel中显示
        if image is not None:
            height, width, channel = image.shape  # 获取图像的高度、宽度和通道数
            bytes_per_line = 3 * width  # 计算图像每行的字节数（对于BGR格式，每个像素3字节）
            # 将BGR格式的OpenCV图像转换为Qt的QImage对象，并交换红色和蓝色通道（BGR -> RGB）
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(1180, 580, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
        
        self.resize(1200, 600)

# --- 后台工作线程 ---

class SearchWorker(QThread):
    """
    后台工作线程，用于执行耗时操作（模型加载、搜索、RANSAC、反馈），避免UI阻塞。
    这是GUI应用中的一个关键设计模式，确保用户界面在执行长时间任务时保持响应。

    信号:
    - model_loaded (bool): 模型加载完成时发出。
    - search_finished (list): 搜索完成时发出，携带结果列表。
    - ransac_finished (list): RANSAC验证完成时发出，携带验证后的结果列表。
    - error_occurred (str): 发生错误时发出，携带错误信息。
    """
    model_loaded = pyqtSignal(bool)  # 定义模型加载完成信号，参数为布尔值（成功/失败）
    search_finished = pyqtSignal(list)  # 定义搜索完成信号，参数为结果列表
    ransac_finished = pyqtSignal(list)  # 定义RANSAC完成信号，参数为重排后的结果列表
    error_occurred = pyqtSignal(str)  # 定义错误发生信号，参数为错误信息字符串

    def __init__(self):
        super().__init__()  # 调用父类构造函数
        # --- 模型和数据 ---
        self.im_features = None  # 存储数据库中所有图像的特征向量
        self.image_paths = None  # 存储数据库中所有图像的文件路径
        self.idf = None  # 存储逆文档频率（Inverse Document Frequency）权重
        self.numWords = None  # 视觉词典中的单词总数
        self.voc_tree = None  # 词汇树对象
        self.inverted_index = None  # 倒排索引数据结构
        self.fea_det = None  # 特征检测器对象（如SIFT）
        self.is_model_loaded = False  # 标志位，表示模型是否已加载
        
        # --- 当前查询状态 ---
        self.query_image_path = None  # 当前查询图像的文件路径
        self.query_vector = None  # 当前查询向量 (可能被相关性反馈更新)
        self.original_query_vector = None # 初始查询向量，用于相关性反馈的基准
        self.query_kpts = None  # 查询图像的关键点 (用于RANSAC)
        self.query_des = None  # 查询图像的描述符 (用于RANSAC)
        self.query_word_indices = None  # 查询图像的单词索引 (用于RANSAC)
        
        # --- 任务控制 ---
        self.action = None  # 字符串标志，用于决定run()方法执行哪个任务 ('load_model', 'search', 'ransac', 'feedback')
        
        # --- 相关性反馈参数 ---
        self.positive_ids = []  # 用户标记为“相关”的图像ID列表
        self.negative_ids = []  # 用户标记为“不相关”的图像ID列表
        
        # --- RANSAC 参数 ---
        self.top_k = 20 # 默认值

    def run(self):
        """
        QThread的核心方法。当调用thread.start()时，此方法会在新线程中被执行。
        根据action标志执行相应的后台任务。
        """
        if self.action == 'load_model':  # 如果任务是加载模型
            self.load_model()
        elif self.action == 'search':  # 如果任务是执行搜索
            self.perform_search()
        elif self.action == 'ransac':  # 如果任务是执行RANSAC
            self.perform_ransac()
        elif self.action == 'feedback':  # 如果任务是执行相关性反馈
            self.perform_feedback()

    def load_model(self):
        """加载预训练的词袋模型文件。"""
        try:
            model_path = "tree-bag-of-words.pkl"  # 模型文件的路径
            if not os.path.exists(model_path):  # 检查模型文件是否存在
                self.error_occurred.emit(f"Model file '{model_path}' not found.")  # 如果不存在，发出错误信号
                return

            # 使用joblib从文件中加载模型数据
            self.im_features, self.image_paths, self.idf, self.numWords, \
            self.voc_tree, self.inverted_index = joblib.load(model_path)
            
            self.fea_det = cv2.SIFT_create()  # 创建SIFT特征检测器实例
            self.is_model_loaded = True  # 更新模型加载状态标志
            self.model_loaded.emit(True)  # 发出模型加载成功的信号
        except Exception as e:
            # 如果加载过程中出现任何异常，发出错误信号
            self.error_occurred.emit(f"Failed to load model: {e}")

    def _extract_query_features(self):
        """提取查询图像的特征，仅在首次搜索或更换查询图片时调用。"""
        img = cv2.imread(self.query_image_path)  # 使用OpenCV读取查询图像
        if img is None:  # 检查图像是否读取成功
            raise ValueError(f"Failed to read image: {self.query_image_path}")

        # 使用SIFT检测器提取关键点和描述符
        self.query_kpts, self.query_des = self.fea_det.detectAndCompute(img, None)
        if self.query_des is None:  # 检查是否检测到任何特征
            raise ValueError("No features detected in the query image.")

        # 使用词汇树将SIFT描述符量化为视觉单词
        words = self.voc_tree.quantize(self.query_des)
        self.query_word_indices = np.array(words, dtype=np.int32)  # 将单词索引转换为NumPy数组

        # 构建该图像的词袋（BoW）向量
        test_features = np.zeros((1, self.numWords), "float32")  # 初始化一个全零向量
        for w in self.query_word_indices:  # 遍历每个视觉单词
            test_features[0][w] += 1  # 在对应单词的维度上加1

        # 应用TF-IDF权重
        test_features = test_features * self.idf
        # 对特征向量进行L2归一化，使其长度为1
        test_features = preprocessing.normalize(test_features, norm='l2')
        
        # 保存初始查询向量和当前查询向量
        self.original_query_vector = test_features.copy()  # 备份原始查询向量
        self.query_vector = test_features  # 设置当前查询向量

    def perform_search(self):
        """执行图像检索。"""
        try:
            # 检查模型是否已加载以及是否已指定查询图片
            if not self.is_model_loaded or not self.query_image_path:
                return

            # 如果是新图片或首次搜索（即查询向量还未计算），则提取特征
            if self.query_vector is None: 
                self._extract_query_features()

            # 使用倒排索引高效地计算查询向量与数据库中所有图像的相似度得分
            score = compute_score_with_inverted_index(self.query_vector, self.inverted_index, len(self.image_paths))
            # 根据得分降序排序，得到图像ID的排名
            rank_ID = np.argsort(-score)

            # 准备并发送结果列表
            results = []
            for i in range(min(self.top_k, len(rank_ID))): # 使用self.top_k获取排名最高的结果
                img_id = rank_ID[i]  # 获取图像ID
                results.append({  # 将结果信息打包成一个字典
                    'id': int(img_id),
                    'path': self.image_paths[img_id],
                    'score': float(score[img_id]),
                    'rank': i + 1
                })
            
            self.search_finished.emit(results)  # 发出搜索完成信号，并附带结果列表

        except Exception as e:
            self.error_occurred.emit(f"Search failed: {e}")  # 如果出错，发出错误信号

    def perform_feedback(self):
        """根据用户反馈更新查询向量并重新搜索。"""
        try:
            # 如果用户没有提供任何反馈，则退化为执行一次标准搜索
            if not self.positive_ids and not self.negative_ids:
                self.perform_search()
                return

            # 从数据库中收集用户标记的正样本和负样本的特征向量
            positive_features = [self.im_features[img_id] for img_id in self.positive_ids if img_id < len(self.im_features)]
            negative_features = [self.im_features[img_id] for img_id in self.negative_ids if img_id < len(self.im_features)]

            # 使用Rocchio算法重新加权查询向量
            self.query_vector = reweight_query_vector(
                self.original_query_vector, # 始终基于原始查询向量进行调整，避免漂移
                np.array(positive_features) if positive_features else np.array([]).reshape(0, self.numWords),
                np.array(negative_features) if negative_features else np.array([]).reshape(0, self.numWords),
                alpha=0.7, beta=0.3, gamma=0.2 # Rocchio算法的参数，控制原始、正、负反馈的权重
            )
            
            # 使用更新后的查询向量执行一次新的搜索
            self.perform_search()
            
        except Exception as e:
            self.error_occurred.emit(f"Relevance feedback failed: {e}")

    def perform_ransac(self):
        """对候选结果执行RANSAC几何验证。"""
        try:
            query_img = cv2.imread(self.query_image_path)  # 读取查询图像
            if query_img is None:
                raise ValueError(f"Failed to read query image: {self.query_image_path}")

            # 从候选结果中提取图像ID
            candidate_ids = [item['id'] for item in self.ransac_candidates]

            # 调用外部的RANSAC验证函数
            verified_results = ransac_verification(
                query_img, self.query_kpts, self.query_des, self.query_word_indices,
                candidate_ids, self.image_paths, self.voc_tree, self.fea_det,
                top_k=len(candidate_ids) # 验证所有提供的候选者
            )

            self.ransac_finished.emit(verified_results)  # 发出RANSAC完成信号，并附带验证后的结果
            
        except Exception as e:
            self.error_occurred.emit(f"RANSAC verification failed: {e}")

# --- 主窗口 ---

class MainWindow(QMainWindow):
    """
    应用程序的主窗口。
    这个类负责管理整个GUI的布局、用户交互和与后台线程的通信。
    """
    def __init__(self):
        super().__init__()  # 调用父类构造函数
        self.setWindowTitle("Image Search Engine")  # 设置窗口标题
        self.resize(1200, 800)  # 设置窗口初始大小

        # 初始化后台工作线程并连接其信号到主窗口的槽函数
        self.worker = SearchWorker()  # 创建工作线程实例
        self.worker.model_loaded.connect(self.on_model_loaded)  # 模型加载信号 -> on_model_loaded槽
        self.worker.search_finished.connect(self.on_search_finished)  # 搜索完成信号 -> on_search_finished槽
        self.worker.ransac_finished.connect(self.on_ransac_finished)  # RANSAC完成信号 -> on_ransac_finished槽
        self.worker.error_occurred.connect(self.on_error)  # 错误信号 -> on_error槽
        
        # 初始化UI
        self.init_ui()
        
        # --- 状态变量 ---
        self.current_image_path = None  # 当前查询图片的文件路径
        self.current_results = []  # 当前显示的搜索结果列表
        self.match_viewer = None # 保持对RANSAC匹配可视化窗口的引用，防止其被Python的垃圾回收机制意外销毁
        
        # 在程序启动时自动开始加载模型
        self._start_model_loading()

    def init_ui(self):
        """初始化主窗口的用户界面。"""
        main_widget = QWidget()  # 创建一个中心部件
        self.setCentralWidget(main_widget)  # 将其设置为主窗口的中心区域
        main_layout = QVBoxLayout(main_widget)  # 创建一个垂直布局管理器

        # 创建并添加UI的各个部分（控制面板、查询面板、结果面板）
        main_layout.addLayout(self._create_control_panel())
        main_layout.addLayout(self._create_query_panel())
        main_layout.addWidget(self._create_results_panel())

        # 设置状态栏
        self.status_label = QLabel("Ready")  # 创建一个标签用于显示状态信息
        self.statusBar().addWidget(self.status_label)  # 将标签添加到主窗口底部的状态栏

    def _create_control_panel(self):
        """创建顶部的控制面板（包含所有按钮和设置）。"""
        top_layout = QHBoxLayout()  # 使用水平布局
        
        # --- 按钮 ---
        self.btn_load = QPushButton("Load Image")  # “加载图片”按钮
        self.btn_load.clicked.connect(self.open_image)  # 连接点击信号到open_image槽函数
        self.btn_load.setEnabled(False)  # 初始时禁用，直到模型加载完成

        self.btn_search = QPushButton("Search")  # “搜索”按钮
        self.btn_search.clicked.connect(self.start_search)  # 连接点击信号到start_search槽函数
        self.btn_search.setEnabled(False)  # 初始时禁用，直到加载了查询图片

        self.btn_feedback = QPushButton("Update (Feedback)")  # “相关性反馈更新”按钮
        self.btn_feedback.clicked.connect(self.start_feedback)  # 连接点击信号到start_feedback槽函数
        self.btn_feedback.setEnabled(False)  # 初始时禁用
        self.btn_feedback.setToolTip("Mark images as Relevant/Irrelevant and click to refine search.")  # 设置鼠标悬停提示

        self.btn_ransac = QPushButton("Verify (RANSAC)")  # “RANSAC验证”按钮
        self.btn_ransac.clicked.connect(self.start_ransac)  # 连接点击信号到start_ransac槽函数
        self.btn_ransac.setEnabled(False)  # 初始时禁用
        self.btn_ransac.setToolTip("Run geometric verification on top results.")  # 设置鼠标悬停提示

        # 将按钮添加到布局中
        top_layout.addWidget(self.btn_load)
        top_layout.addWidget(self.btn_search)
        top_layout.addWidget(self.btn_feedback)
        top_layout.addWidget(self.btn_ransac)
        
        # --- RANSAC Top-K 设置 ---
        # --- Search Top-K 设置 ---
        top_layout.addWidget(QLabel("Search Top-K:"))  # 添加一个标签
        self.spin_search_k = QSpinBox()  # 创建一个数字输入框用于控制搜索结果数量
        self.spin_search_k.setRange(5, 20)  # 设置允许输入的范围，例如10到100
        self.spin_search_k.setValue(10)  # 设置默认显示20个结果
        self.spin_search_k.setToolTip("Set the number of top results to retrieve.") # 设置鼠标悬停提示
        top_layout.addWidget(self.spin_search_k)  # 添加到布局

        # --- RANSAC Top-K 设置 ---
        top_layout.addWidget(QLabel("RANSAC Top-K:"))  # 添加一个标签
        self.spin_ransac_k = QSpinBox()  # 创建一个数字输入框
        self.spin_ransac_k.setRange(3, 10)  # 设置允许输入的范围
        self.spin_ransac_k.setValue(3)  # 设置默认值
        top_layout.addWidget(self.spin_ransac_k)  # 添加到布局
        
        top_layout.addStretch()  # 添加一个伸缩弹簧，使所有控件靠左对齐
        return top_layout  # 返回创建好的布局

    def _create_query_panel(self):
        """创建查询图片显示区域。"""
        display_layout = QHBoxLayout()  # 使用水平布局
        query_group = QGroupBox("Query Image")  # 创建一个带标题的分组框
        query_layout = QVBoxLayout(query_group)  # 在分组框内部使用垂直布局
        
        self.query_label = QLabel("Load an image to start")  # 创建用于显示查询图片的标签
        self.query_label.setAlignment(Qt.AlignCenter)  # 居中对齐
        self.query_label.setFixedSize(320, 240)  # 设置固定大小
        self.query_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")  # 设置样式
        
        query_layout.addWidget(self.query_label)  # 将标签添加到分组框布局中
        display_layout.addWidget(query_group)  # 将分组框添加到外部布局中
        display_layout.addStretch()  # 添加伸缩弹簧
        return display_layout  # 返回创建好的布局

    def _create_results_panel(self):
        """创建结果展示区域（一个可滚动的网格）。"""
        self.scroll_area = QScrollArea()  # 创建滚动区域
        self.scroll_area.setWidgetResizable(True)  # 允许内部部件调整大小
        self.results_widget = QWidget()  # 创建一个容器部件，将放在滚动区域内
        self.results_grid = QGridLayout(self.results_widget)  # 在容器部件上创建一个网格布局
        self.scroll_area.setWidget(self.results_widget)  # 将容器部件设置到滚动区域中
        return self.scroll_area  # 返回创建好的滚动区域

    def _start_model_loading(self):
        """在后台启动模型加载过程。"""
        self.status_label.setText("Loading model... Please wait.")  # 更新状态栏信息
        self.worker.action = 'load_model'  # 告诉后台线程要执行“加载模型”任务
        self.worker.start()  # 启动线程

    # --- UI交互和槽函数 ---

    def open_image(self):
        """打开文件对话框以选择查询图片。"""
        # 弹出文件选择对话框，只显示指定类型的图像文件
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:  # 如果用户成功选择了一个文件
            self.current_image_path = file_path  # 保存文件路径
            self.show_image_on_label(file_path, self.query_label)  # 在查询区域显示图片
            self.btn_search.setEnabled(True)  # 启用“搜索”按钮
            self.status_label.setText(f"Loaded: {os.path.basename(file_path)}")  # 更新状态栏
            
            # 重置状态以进行一次全新的搜索
            self.clear_results()  # 清空上一次的搜索结果
            self.btn_feedback.setEnabled(False)  # 禁用反馈按钮
            self.btn_ransac.setEnabled(False)  # 禁用RANSAC按钮
            self.worker.query_vector = None # 强制后台线程在下次搜索时重新提取特征

    def show_image_on_label(self, image_path, label_widget):
        """在指定的QLabel上显示图片。"""
        pixmap = QPixmap(image_path)  # 从文件路径加载图片为QPixmap
        if not pixmap.isNull():  # 检查图片是否加载成功
            # 将图片缩放到与标签大小一致，同时保持宽高比
            scaled_pixmap = pixmap.scaled(label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label_widget.setPixmap(scaled_pixmap)  # 在标签上显示缩放后的图片
        else:
            label_widget.setText("Failed to load image")  # 如果加载失败，显示错误文本

    def start_search(self):
        """开始执行搜索。这是“Search”按钮的槽函数。"""
        if not self.current_image_path:  # 如果没有加载查询图片，则不执行任何操作
            return
        
        self.status_label.setText("Searching...")  # 更新状态栏
        self.set_controls_enabled(False)  # 禁用所有控制按钮，防止用户在搜索期间进行其他操作

        self.worker.top_k = self.spin_search_k.value() # 传递Top-K值
        self.worker.query_image_path = self.current_image_path  # 将查询图片路径传递给后台线程
        self.worker.action = 'search'  # 告诉后台线程执行“搜索”任务
        self.worker.start()  # 启动线程

    def start_feedback(self):
        """开始执行相关性反馈。这是“Update (Feedback)”按钮的槽函数。"""
        self.status_label.setText("Updating query with feedback...")  # 更新状态栏
        self.set_controls_enabled(False)  # 禁用控制按钮
        
        # 从UI收集用户标记为“相关”和“不相关”的图像ID
        positive_ids = []
        negative_ids = []
        for i in range(self.results_grid.count()):  # 遍历结果网格中的每一个项目
            widget = self.results_grid.itemAt(i).widget()  # 获取项目中的QWidget
            if widget:
                # 查找子控件中的复选框
                chk_pos = widget.findChild(QCheckBox, "chk_pos")
                chk_neg = widget.findChild(QCheckBox, "chk_neg")
                img_id = widget.property("img_id")  # 获取之前存储在控件上的图像ID
                
                if chk_pos and chk_pos.isChecked():  # 如果“相关”复选框被选中
                    positive_ids.append(img_id)
                if chk_neg and chk_neg.isChecked():  # 如果“不相关”复选框被选中
                    negative_ids.append(img_id)
        
        # 将收集到的ID列表传递给后台线程
        self.worker.positive_ids = positive_ids
        self.worker.negative_ids = negative_ids
        self.worker.action = 'feedback'  # 告诉后台线程执行“反馈”任务
        self.worker.start()  # 启动线程

    def start_ransac(self):
        """开始执行RANSAC验证。这是“Verify (RANSAC)”按钮的槽函数。"""
        if not self.current_results:  # 如果没有搜索结果，则不执行
            return

        self.status_label.setText("Verifying with RANSAC...")  # 更新状态栏
        self.set_controls_enabled(False)  # 禁用控制按钮
        
        top_k = self.spin_ransac_k.value()  # 从数字输入框获取要验证的结果数量
        self.worker.ransac_candidates = self.current_results[:top_k]  # 将排名前k的结果作为候选者传递给后台线程
        self.worker.action = 'ransac'  # 告诉后台线程执行“RANSAC”任务
        self.worker.start()  # 启动线程

    def set_controls_enabled(self, enabled):
        """统一启用或禁用所有控制按钮。"""
        self.btn_load.setEnabled(enabled)
        self.btn_search.setEnabled(enabled)
        # 反馈和RANSAC按钮只有在有结果时才应启用
        self.btn_feedback.setEnabled(enabled and bool(self.current_results))
        self.btn_ransac.setEnabled(enabled and bool(self.current_results))

    def show_match_viewer(self, vis_image, title):
        """创建并显示RANSAC匹配的可视化窗口。"""
        # 创建新窗口实例并将其保存为类成员，以防止它被垃圾回收
        self.match_viewer = MatchViewerWindow(vis_image, title)
        self.match_viewer.show()  # 显示窗口

    # --- Worker信号的槽函数 ---

    def on_model_loaded(self, success):
        """模型加载完成后的回调函数。"""
        if success:  # 如果加载成功
            self.status_label.setText("Model loaded. Ready to search.")  # 更新状态栏
            self.btn_load.setEnabled(True)  # 启用“加载图片”按钮
        else:  # 如果加载失败
            self.status_label.setText("Failed to load model.")

    def on_search_finished(self, results):
        """搜索完成后的回调函数。"""
        self.current_results = results  # 保存当前结果
        self.status_label.setText(f"Search finished. Found {len(results)} results.")  # 更新状态栏
        self.set_controls_enabled(True)  # 重新启用控制按钮
        self.display_results(results, mode='search')  # 显示搜索结果

    def on_ransac_finished(self, results):
        """RANSAC验证完成后的回调函数。"""
        self.current_results = results  # 保存当前结果
        self.status_label.setText(f"RANSAC finished. {len(results)} matches found.")  # 更新状态栏
        self.set_controls_enabled(True)  # 重新启用控制按钮
        self.display_results(results, mode='ransac')  # 显示RANSAC重排后的结果

        # # 自动显示 top 1 结果的匹配可视化，如果存在
        # if results and 'vis' in results[0] and results[0]['vis'] is not None:
        #     self.show_match_viewer(results[0]['vis'], f"Top Match with Image {results[0]['doc_id']}")

    def on_error(self, message):
        """发生错误时的回调函数。"""
        QMessageBox.critical(self, "Error", message)  # 弹出一个严重错误对话框显示错误信息
        self.status_label.setText("Error occurred.")  # 更新状态栏
        self.set_controls_enabled(True)  # 重新启用控制按钮，以便用户可以尝试其他操作

    # --- 结果显示 ---

    def clear_results(self):
        """清除结果网格中的所有项目。"""
        while self.results_grid.count():  # 当网格中还有项目时循环
            item = self.results_grid.takeAt(0)  # 取出第一个项目
            if item and item.widget():  # 如果项目存在且包含一个QWidget
                item.widget().deleteLater()  # 安全地删除该QWidget，释放内存

    def display_results(self, results, mode='search'):
        """在网格中显示结果列表。"""
        self.clear_results()  # 首先清空旧的结果
        
        cols = 4 # 设置每行显示4个结果
        top_k = self.spin_search_k.value()
        for idx, result in enumerate(results):
            if idx >= top_k: break # 最多只显示top_k个结果

            row, col = idx // cols, idx % cols  # 计算项目在网格中的行和列
            
            # 创建并添加单个结果项的QWidget
            item_widget = self._create_result_item_widget(result, mode)
            self.results_grid.addWidget(item_widget, row, col)  # 将其添加到网格布局中

    def _create_result_item_widget(self, result, mode):
        """创建单个结果项的QWidget。这是一个辅助方法，用于减少display_results中的代码重复。"""
        item_widget = ClickableQWidget()  # 使用我们自定义的可点击QWidget
        item_layout = QVBoxLayout(item_widget)  # 在其中使用垂直布局
        item_layout.setContentsMargins(5, 5, 5, 5)  # 设置边距

        # --- 图片标签 ---
        img_label = QLabel()  # 创建用于显示结果图片的标签
        img_label.setFixedSize(200, 150)  # 设置固定大小
        img_label.setAlignment(Qt.AlignCenter)  # 居中对齐
        img_label.setStyleSheet("border: 1px solid #ddd;")  # 设置边框样式
        
        pixmap = QPixmap(result['path'])  # 从路径加载图片
        if not pixmap.isNull():  # 如果加载成功
            # 缩放并显示图片
            img_label.setPixmap(pixmap.scaled(img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            img_label.setText("Image not found")  # 如果失败，显示提示文本
        
        item_layout.addWidget(img_label)  # 将图片标签添加到布局

        # --- 信息标签和交互控件 ---
        if mode == 'search':  # 如果是标准搜索模式
            item_widget.setProperty("img_id", result['id'])  # 将图像ID存储在控件的属性中，以便后续引用
            
            # 创建一个水平布局用于放置信息和复选框
            info_layout = QHBoxLayout()
            info_label = QLabel(f"Rank: {result['rank']}, Score: {result['score']:.2f}")  # 显示排名和得分
            info_layout.addWidget(info_label)
            info_layout.addStretch()
            
            # 创建相关性反馈的复选框
            chk_pos = QCheckBox("Rel")  # "Relevant" (相关)
            chk_pos.setObjectName("chk_pos") # 设置对象名，方便之后查找
            chk_neg = QCheckBox("Irrel") # "Irrelevant" (不相关)
            chk_neg.setObjectName("chk_neg")
            
            info_layout.addWidget(chk_pos)
            info_layout.addWidget(chk_neg)
            item_layout.addLayout(info_layout)

        elif mode == 'ransac':  # 如果是RANSAC模式
            item_widget.setProperty("img_id", result['doc_id'])  # RANSAC结果使用'doc_id'
            info_label = QLabel(f"Inliers: {result['inliers']}")  # 显示内点数量
            item_layout.addWidget(info_label)
            
            # 如果有可视化图像，使整个控件可点击以显示它
            if 'vis' in result and result['vis'] is not None:
                # 使用lambda函数捕获必要的参数，连接点击信号到槽函数
                item_widget.clicked.connect(
                    lambda vis_img=result['vis'], doc_id=result['doc_id']: 
                    self.show_match_viewer(vis_img, f"Match with Image {doc_id}")
                )
                item_widget.setToolTip("Click to see feature matches") # 添加鼠标悬停提示

        return item_widget  # 返回创建好的结果项控件

# --- 程序入口 ---

if __name__ == '__main__':
    app = QApplication(sys.argv)  # 创建QApplication实例
    main_win = MainWindow()  # 创建主窗口实例
    main_win.show()  # 显示主窗口
    sys.exit(app.exec_())  # 进入应用程序的事件循环，并确保在退出时返回正确的状态码
