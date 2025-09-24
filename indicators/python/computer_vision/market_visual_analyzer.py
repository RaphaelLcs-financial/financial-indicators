"""
计算机视觉市场分析指标
Computer Vision Market Analysis Indicators

基于深度学习的市场图表模式识别和视觉分析系统
支持技术形态识别、趋势线检测和模式预测
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from PIL import Image, ImageDraw, ImageFont
    TORCH_AVAILABLE = True
    print("🧠 PyTorch 已启用")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch 不可用")

# 传统计算机视觉支持
try:
    import cv2
    from skimage import measure, feature, filters
    from skimage.transform import resize, hough_line, hough_circle, hough_ellipse
    from scipy import ndimage, signal
    CV_AVAILABLE = True
    print("👁️ OpenCV 已启用")
except ImportError:
    CV_AVAILABLE = False
    print("⚠️ OpenCV 不可用")

class MarketVisualAnalyzer:
    """
    市场视觉分析器

    基于计算机视觉技术分析金融图表
    支持技术模式识别、趋势线检测、形态分析
    """

    def __init__(self, image_size: Tuple[int, int] = (224, 224),
                 pattern_types: List[str] = None,
                 detection_threshold: float = 0.7):
        """
        初始化市场视觉分析器

        Args:
            image_size: 图像尺寸
            pattern_types: 识别的模式类型
            detection_threshold: 检测阈值
        """
        self.image_size = image_size
        self.pattern_types = pattern_types or [
            'head_and_shoulders', 'double_top', 'double_bottom',
            'triangle', 'wedge', 'flag', 'pennant', 'cup_and_handle'
        ]
        self.detection_threshold = detection_threshold

        # 深度学习模型
        self.pattern_recognition_model = None
        self.trend_detection_model = None
        self.feature_extractor = None

        # 传统视觉算法
        self.edge_detector = None
        self.line_detector = None
        self.circle_detector = None

        # 分析结果
        self.detected_patterns = []
        self.trend_lines = []
        self.support_resistance_levels = []
        self.visual_features = {}

        # 性能指标
        self.detection_stats = {
            'patterns_detected': 0,
            'trend_lines_found': 0,
            'support_resistance_levels': 0,
            'detection_accuracy': 0.0
        }

    def initialize_models(self):
        """初始化计算机视觉模型"""
        if TORCH_AVAILABLE:
            try:
                # 初始化模式识别模型
                self.pattern_recognition_model = self._create_pattern_recognition_model()

                # 初始化特征提取器
                self.feature_extractor = self._create_feature_extractor()

                print("✅ 深度学习视觉模型初始化成功")
            except Exception as e:
                print(f"⚠️ 深度学习模型初始化失败: {e}")

        if CV_AVAILABLE:
            try:
                # 初始化传统视觉算法
                self.edge_detector = cv2.Canny
                self.line_detector = cv2.HoughLinesP
                self.circle_detector = cv2.HoughCircles

                print("✅ 传统视觉算法初始化成功")
            except Exception as e:
                print(f"⚠️ 传统视觉算法初始化失败: {e}")

    def _create_pattern_recognition_model(self) -> nn.Module:
        """创建模式识别模型"""
        class PatternRecognitionNet(nn.Module):
            def __init__(self, num_patterns: int = 8):
                super().__init__()

                # CNN特征提取
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(32),
                    nn.MaxPool2d(2),

                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(64),
                    nn.MaxPool2d(2),

                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(128),
                    nn.MaxPool2d(2),

                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(256),
                    nn.AdaptiveAvgPool2d((1, 1))
                )

                # 分类头
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_patterns)
                )

            def forward(self, x):
                features = self.conv_layers(x)
                features = features.view(features.size(0), -1)
                output = self.classifier(features)
                return output

        return PatternRecognitionNet(len(self.pattern_types))

    def _create_feature_extractor(self) -> nn.Module:
        """创建特征提取器"""
        class FeatureExtractor(nn.Module):
            def __init__(self):
                super().__init__()

                self.conv_layers = nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.BatchNorm2d(64),
                    nn.MaxPool2d(3, stride=2, padding=1),

                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(128),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(128),
                    nn.MaxPool2d(2),

                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(256),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(256),
                    nn.AdaptiveAvgPool2d((1, 1))
                )

            def forward(self, x):
                return self.conv_layers(x)

        return FeatureExtractor()

    def create_chart_image(self, market_data: pd.DataFrame,
                         chart_type: str = 'candlestick',
                         time_period: int = 100) -> np.ndarray:
        """创建市场图表图像"""
        print(f"📊 创建{chart_type}图表图像...")

        # 获取最近的数据
        recent_data = market_data.tail(time_period).copy()

        if 'close' not in recent_data.columns:
            raise ValueError("市场数据必须包含 'close' 列")

        # 标准化价格数据
        min_price = recent_data['close'].min()
        max_price = recent_data['close'].max()
        price_range = max_price - min_price

        if price_range == 0:
            price_range = 1

        normalized_prices = (recent_data['close'] - min_price) / price_range

        # 创建图像
        if chart_type == 'line':
            image = self._create_line_chart(normalized_prices)
        elif chart_type == 'candlestick':
            image = self._create_candlestick_chart(recent_data)
        elif chart_type == 'volume_profile':
            image = self._create_volume_profile_chart(recent_data)
        else:
            image = self._create_line_chart(normalized_prices)

        return image

    def _create_line_chart(self, prices: pd.Series) -> np.ndarray:
        """创建线性图表"""
        height, width = self.image_size
        image = np.ones((height, width), dtype=np.uint8) * 255

        # 绘制价格线
        x_coords = np.linspace(10, width-10, len(prices))
        y_coords = (1 - prices.values) * (height - 20) + 10

        # 转换为整数坐标
        points = np.column_stack([x_coords.astype(int), y_coords.astype(int)])

        # 绘制线条
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i+1], 0, 2)

        return image

    def _create_candlestick_chart(self, data: pd.DataFrame) -> np.ndarray:
        """创建K线图"""
        height, width = self.image_size
        image = np.ones((height, width), dtype=np.uint8) * 255

        # 计算蜡烛参数
        n_candles = len(data)
        candle_width = max(1, (width - 20) // n_candles)

        # 标准化价格
        min_price = data[['high', 'low', 'open', 'close']].min().min()
        max_price = data[['high', 'low', 'open', 'close']].max().max()
        price_range = max_price - min_price

        if price_range == 0:
            price_range = 1

        for i, (_, row) in enumerate(data.iterrows()):
            x = 10 + i * candle_width

            # 计算价格位置
            open_y = int((1 - (row['open'] - min_price) / price_range) * (height - 20) + 10)
            close_y = int((1 - (row['close'] - min_price) / price_range) * (height - 20) + 10)
            high_y = int((1 - (row['high'] - min_price) / price_range) * (height - 20) + 10)
            low_y = int((1 - (row['low'] - min_price) / price_range) * (height - 20) + 10)

            # 确定颜色（上涨为白色，下跌为黑色）
            color = 255 if row['close'] >= row['open'] else 0

            # 绘制蜡烛
            cv2.line(image, (x + candle_width//2, high_y), (x + candle_width//2, low_y), color, 1)

            if candle_width > 2:
                cv2.rectangle(image, (x, min(open_y, close_y)),
                             (x + candle_width, max(open_y, close_y)), color, -1)

        return image

    def _create_volume_profile_chart(self, data: pd.DataFrame) -> np.ndarray:
        """创建成交量分布图"""
        height, width = self.image_size
        image = np.ones((height, width), dtype=np.uint8) * 255

        if 'volume' not in data.columns:
            return image

        # 标准化成交量
        volumes = data['volume'].values
        max_volume = volumes.max()
        normalized_volumes = volumes / max_volume if max_volume > 0 else volumes

        # 绘制成交量柱状图
        bar_width = max(1, (width - 20) // len(volumes))

        for i, volume in enumerate(normalized_volumes):
            x = 10 + i * bar_width
            bar_height = int(volume * (height - 20))
            y = height - 10 - bar_height

            cv2.rectangle(image, (x, y), (x + bar_width, height - 10), 128, -1)

        return image

    def detect_chart_patterns(self, chart_image: np.ndarray) -> Dict[str, Any]:
        """检测图表模式"""
        print("🔍 检测图表模式...")

        pattern_results = {
            'detected_patterns': [],
            'pattern_confidences': {},
            'pattern_locations': {},
            'visual_features': {}
        }

        # 深度学习方法
        if TORCH_AVAILABLE and self.pattern_recognition_model is not None:
            dl_patterns = self._detect_patterns_deep_learning(chart_image)
            pattern_results['detected_patterns'].extend(dl_patterns['patterns'])
            pattern_results['pattern_confidences'].update(dl_patterns['confidences'])

        # 传统计算机视觉方法
        if CV_AVAILABLE:
            cv_patterns = self._detect_patterns_computer_vision(chart_image)
            pattern_results['detected_patterns'].extend(cv_patterns['patterns'])
            pattern_results['pattern_locations'].update(cv_patterns['locations'])

        # 提取视觉特征
        visual_features = self._extract_visual_features(chart_image)
        pattern_results['visual_features'] = visual_features

        # 去重
        pattern_results['detected_patterns'] = list(set(pattern_results['detected_patterns']))

        # 更新统计信息
        self.detection_stats['patterns_detected'] = len(pattern_results['detected_patterns'])

        return pattern_results

    def _detect_patterns_deep_learning(self, image: np.ndarray) -> Dict[str, Any]:
        """基于深度学习的模式检测"""
        if not TORCH_AVAILABLE or self.pattern_recognition_model is None:
            return {'patterns': [], 'confidences': {}}

        try:
            # 预处理图像
            tensor_image = self._preprocess_image_for_dl(image)

            # 模型推理
            with torch.no_grad():
                outputs = self.pattern_recognition_model(tensor_image)
                probabilities = F.softmax(outputs, dim=1)
                predictions = probabilities.squeeze().numpy()

            # 获取检测结果
            detected_patterns = []
            confidences = {}

            for i, (pattern_type, confidence) in enumerate(zip(self.pattern_types, predictions)):
                if confidence > self.detection_threshold:
                    detected_patterns.append(pattern_type)
                    confidences[pattern_type] = float(confidence)

            return {'patterns': detected_patterns, 'confidences': confidences}

        except Exception as e:
            print(f"深度学习模式检测失败: {e}")
            return {'patterns': [], 'confidences': {}}

    def _detect_patterns_computer_vision(self, image: np.ndarray) -> Dict[str, Any]:
        """基于传统计算机视觉的模式检测"""
        if not CV_AVAILABLE:
            return {'patterns': [], 'locations': {}}

        try:
            # 边缘检测
            edges = cv2.Canny(image, 50, 150)

            # 直线检测
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                   minLineLength=30, maxLineGap=10)

            detected_patterns = []
            locations = {}

            if lines is not None:
                # 分析线条模式
                line_patterns = self._analyze_line_patterns(lines, image.shape)
                detected_patterns.extend(line_patterns['patterns'])
                locations.update(line_patterns['locations'])

            # 几何形状检测
            geometric_patterns = self._detect_geometric_shapes(image)
            detected_patterns.extend(geometric_patterns['patterns'])
            locations.update(geometric_patterns['locations'])

            return {'patterns': detected_patterns, 'locations': locations}

        except Exception as e:
            print(f"传统视觉模式检测失败: {e}")
            return {'patterns': [], 'locations': {}}

    def _analyze_line_patterns(self, lines, image_shape) -> Dict[str, Any]:
        """分析线条模式"""
        patterns = []
        locations = {}

        if lines is None:
            return {'patterns': [], 'locations': {}}

        # 分析线条角度和长度
        angles = []
        lengths = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            angles.append(angle)
            lengths.append(length)

        # 检测趋势线模式
        if len(angles) > 3:
            # 检测支撑线和阻力线
            horizontal_lines = [angle for angle in angles if abs(angle) < 10 or abs(angle) > 170]
            if len(horizontal_lines) > len(angles) * 0.3:
                patterns.append('support_resistance')
                locations['support_resistance'] = {'count': len(horizontal_lines)}

            # 检测趋势通道
            angle_std = np.std(angles)
            if angle_std < 20:  # 角度一致性高
                avg_angle = np.mean(angles)
                if -30 < avg_angle < 30:  # 近似水平
                    patterns.append('horizontal_channel')
                elif avg_angle > 30:
                    patterns.append('ascending_channel')
                elif avg_angle < -30:
                    patterns.append('descending_channel')

                locations['channel'] = {'angle': avg_angle, 'consistency': 1.0 - angle_std/45}

        return {'patterns': patterns, 'locations': locations}

    def _detect_geometric_shapes(self, image: np.ndarray) -> Dict[str, Any]:
        """检测几何形状"""
        patterns = []
        locations = {}

        try:
            # 圆形检测（用于检测底部形态）
            circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20,
                                     param1=50, param2=30, minRadius=10, maxRadius=100)

            if circles is not None:
                patterns.append('rounding_bottom')
                locations['rounding_bottom'] = {'count': len(circles[0])}

            # 三角形检测
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) >= 3:
                    # 近似多边形
                    epsilon = 0.04 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    if len(approx) == 3:
                        patterns.append('triangle')
                        locations['triangle'] = {'vertices': 3, 'area': cv2.contourArea(contour)}
                    elif len(approx) == 4:
                        patterns.append('rectangle')
                        locations['rectangle'] = {'vertices': 4, 'area': cv2.contourArea(contour)}

        except Exception as e:
            print(f"几何形状检测失败: {e}")

        return {'patterns': patterns, 'locations': locations}

    def _extract_visual_features(self, image: np.ndarray) -> Dict[str, Any]:
        """提取视觉特征"""
        features = {}

        # 基础统计特征
        features['mean_intensity'] = np.mean(image)
        features['std_intensity'] = np.std(image)
        features['entropy'] = self._calculate_entropy(image)

        # 纹理特征
        features['texture_energy'] = self._calculate_texture_energy(image)
        features['texture_contrast'] = self._calculate_texture_contrast(image)

        # 形状特征
        features['aspect_ratio'] = image.shape[1] / image.shape[0]
        features['center_of_mass'] = ndimage.center_of_mass(image)

        # 频域特征
        features['dominant_frequency'] = self._calculate_dominant_frequency(image)

        return features

    def _calculate_entropy(self, image: np.ndarray) -> float:
        """计算图像熵"""
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram = histogram / histogram.sum()
        entropy = -np.sum(histogram * np.log2(histogram + 1e-10))
        return entropy

    def _calculate_texture_energy(self, image: np.ndarray) -> float:
        """计算纹理能量"""
        # 使用灰度共生矩阵的简化版本
        return np.sum(image**2) / image.size

    def _calculate_texture_contrast(self, image: np.ndarray) -> float:
        """计算纹理对比度"""
        return np.std(image)

    def _calculate_dominant_frequency(self, image: np.ndarray) -> float:
        """计算主频率"""
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        # 找到主频率
        max_idx = np.unravel_index(np.argmax(magnitude_spectrum), magnitude_spectrum.shape)
        return np.sqrt(max_idx[0]**2 + max_idx[1]**2)

    def _preprocess_image_for_dl(self, image: np.ndarray) -> torch.Tensor:
        """为深度学习预处理图像"""
        # 调整大小
        if image.shape != self.image_size:
            image = cv2.resize(image, self.image_size)

        # 归一化
        image = image.astype(np.float32) / 255.0

        # 添加通道维度
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)

        # 转换为tensor
        tensor = torch.FloatTensor(image).unsqueeze(0)

        return tensor

    def detect_trend_lines(self, chart_image: np.ndarray) -> Dict[str, Any]:
        """检测趋势线"""
        print("📈 检测趋势线...")

        if not CV_AVAILABLE:
            return {'trend_lines': [], 'support_resistance': []}

        try:
            # 边缘检测
            edges = cv2.Canny(chart_image, 50, 150)

            # 直线检测
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                   minLineLength=30, maxLineGap=10)

            trend_lines = []
            support_resistance = []

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]

                    # 计算线条属性
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                    line_info = {
                        'start': (x1, y1),
                        'end': (x2, y2),
                        'angle': angle,
                        'length': length,
                        'type': self._classify_trend_line(angle)
                    }

                    # 分类趋势线
                    if line_info['type'] == 'support':
                        support_resistance.append(line_info)
                    elif line_info['type'] == 'resistance':
                        support_resistance.append(line_info)
                    else:
                        trend_lines.append(line_info)

            # 更新统计信息
            self.detection_stats['trend_lines_found'] = len(trend_lines)
            self.detection_stats['support_resistance_levels'] = len(support_resistance)

            return {
                'trend_lines': trend_lines,
                'support_resistance': support_resistance
            }

        except Exception as e:
            print(f"趋势线检测失败: {e}")
            return {'trend_lines': [], 'support_resistance': []}

    def _classify_trend_line(self, angle: float) -> str:
        """分类趋势线"""
        # 标准化角度到[-90, 90]
        while angle > 90:
            angle -= 180
        while angle < -90:
            angle += 180

        if abs(angle) < 15:
            return 'support_resistance'
        elif angle > 15:
            return 'resistance'
        elif angle < -15:
            return 'support'
        else:
            return 'neutral'

    def generate_visual_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """生成视觉交易信号"""
        print("🎯 生成视觉交易信号...")

        # 创建图表图像
        chart_image = self.create_chart_image(market_data)

        # 检测模式
        patterns = self.detect_chart_patterns(chart_image)

        # 检测趋势线
        trend_lines = self.detect_trend_lines(chart_image)

        # 生成信号
        signals = []
        confidences = []
        pattern_signals = []

        for i in range(len(market_data)):
            # 基于模式生成信号
            pattern_signal = self._generate_pattern_signal(patterns['detected_patterns'])
            pattern_signals.append(pattern_signal)

            # 基于趋势线生成信号
            trend_signal = self._generate_trend_signal(trend_lines)

            # 综合信号
            combined_signal = 0.6 * pattern_signal + 0.4 * trend_signal
            signal = np.tanh(combined_signal)

            # 置信度
            confidence = self._calculate_visual_confidence(patterns, trend_lines)

            signals.append(signal)
            confidences.append(confidence)

        signals_df = pd.DataFrame({
            'visual_signal': signals,
            'confidence': confidences,
            'pattern_count': [len(patterns['detected_patterns'])] * len(signals),
            'trend_line_count': [len(trend_lines['trend_lines'])] * len(signals),
            'support_resistance_count': [len(trend_lines['support_resistance'])] * len(signals)
        }, index=market_data.index)

        return signals_df

    def _generate_pattern_signal(self, detected_patterns: List[str]) -> float:
        """基于模式生成信号"""
        pattern_signals = {
            'head_and_shoulders': -0.8,
            'inverse_head_and_shoulders': 0.8,
            'double_top': -0.7,
            'double_bottom': 0.7,
            'triangle': 0.3,
            'ascending_triangle': 0.6,
            'descending_triangle': -0.6,
            'flag': 0.4,
            'pennant': 0.4,
            'cup_and_handle': 0.7,
            'rounding_bottom': 0.6
        }

        signal = 0.0
        for pattern in detected_patterns:
            if pattern in pattern_signals:
                signal += pattern_signals[pattern]

        return np.clip(signal, -1.0, 1.0)

    def _generate_trend_signal(self, trend_lines: Dict) -> float:
        """基于趋势线生成信号"""
        signal = 0.0

        # 分析趋势线
        up_trends = [line for line in trend_lines['trend_lines'] if line['angle'] > 10]
        down_trends = [line for line in trend_lines['trend_lines'] if line['angle'] < -10]

        if len(up_trends) > len(down_trends):
            signal = 0.5
        elif len(down_trends) > len(up_trends):
            signal = -0.5

        return signal

    def _calculate_visual_confidence(self, patterns: Dict, trend_lines: Dict) -> float:
        """计算视觉置信度"""
        confidence = 0.0

        # 基于检测到的模式数量
        pattern_count = len(patterns['detected_patterns'])
        confidence += min(pattern_count * 0.2, 0.6)

        # 基于趋势线置信度
        trend_count = len(trend_lines['trend_lines'])
        confidence += min(trend_count * 0.1, 0.3)

        # 基于支撑阻力位
        sr_count = len(trend_lines['support_resistance'])
        confidence += min(sr_count * 0.1, 0.1)

        return min(confidence, 1.0)

    def get_visual_insights(self) -> Dict[str, Any]:
        """获取视觉分析洞察"""
        return {
            'image_size': self.image_size,
            'pattern_types': self.pattern_types,
            'detection_threshold': self.detection_threshold,
            'detection_stats': self.detection_stats,
            'recent_patterns': self.detected_patterns[-5:],  # 最近5个模式
            'visual_features': self.visual_features,
            'torch_available': TORCH_AVAILABLE,
            'cv_available': CV_AVAILABLE
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'image_size': self.image_size,
            'pattern_types': self.pattern_types,
            'detection_threshold': self.detection_threshold,
            'detection_stats': self.detection_stats,
            'models_initialized': self.pattern_recognition_model is not None,
            'torch_available': TORCH_AVAILABLE,
            'cv_available': CV_AVAILABLE,
            'model_type': 'Computer Vision Market Analyzer'
        }

# 便捷函数
def create_market_visual_analyzer(image_size: Tuple[int, int] = (224, 224),
                                pattern_types: List[str] = None,
                                detection_threshold: float = 0.7) -> MarketVisualAnalyzer:
    """创建市场视觉分析器实例"""
    analyzer = MarketVisualAnalyzer(image_size, pattern_types, detection_threshold)
    analyzer.initialize_models()
    return analyzer

def quick_visual_analysis(market_data: pd.DataFrame) -> Dict[str, Any]:
    """快速视觉分析"""
    analyzer = MarketVisualAnalyzer()
    analyzer.initialize_models()

    # 创建图表
    chart_image = analyzer.create_chart_image(market_data)

    # 检测模式
    patterns = analyzer.detect_chart_patterns(chart_image)

    # 生成信号
    signals = analyzer.generate_visual_signals(market_data)

    return {
        'detected_patterns': patterns['detected_patterns'],
        'latest_signal': signals['visual_signal'].iloc[-1] if len(signals) > 0 else 0,
        'average_confidence': signals['confidence'].mean() if len(signals) > 0 else 0,
        'model_info': analyzer.get_model_info()
    }