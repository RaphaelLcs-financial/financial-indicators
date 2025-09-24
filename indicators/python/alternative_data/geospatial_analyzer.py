"""
地理空间分析器金融指标

本模块实现了基于地理空间数据的另类金融指标，包括卫星图像、人流数据、地理位置分析等。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy import stats
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')


class GeospatialAnalyzer:
    """
    地理空间分析器

    分析基于地理位置和空间分布的另类数据指标
    """

    def __init__(self,
                 spatial_window: int = 30,
                 distance_threshold: float = 100,  # 公里
                 clustering_threshold: float = 0.7):
        """
        初始化地理空间分析器

        Args:
            spatial_window: 空间分析窗口
            distance_threshold: 距离阈值（公里）
            clustering_threshold: 聚类阈值
        """
        self.spatial_window = spatial_window
        self.distance_threshold = distance_threshold
        self.clustering_threshold = clustering_threshold

        # 地理坐标缓存
        self.location_cache = {}
        self.spatial_patterns = {}

    def calculate_satellite_imagery_metrics(self, satellite_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算卫星图像指标

        Args:
            satellite_data: 卫星图像数据

        Returns:
            卫星图像指标字典
        """
        metrics = {}

        # 夜间灯光指数
        if 'night_light_intensity' in satellite_data.columns:
            night_light = satellite_data['night_light_intensity']
            metrics['night_light_ma'] = night_light.rolling(self.spatial_window).mean()
            metrics['night_light_growth'] = night_light.pct_change().rolling(self.spatial_window).mean()

            # 灯光密度
            metrics['light_density'] = night_light.rolling(self.spatial_window).apply(
                lambda x: np.sum(x > np.percentile(x, 80)) / len(x)
            )

        # 建筑活动指标
        if 'construction_activity' in satellite_data.columns:
            construction = satellite_data['construction_activity']
            metrics['construction_intensity'] = construction.rolling(self.spatial_window).mean()
            metrics['construction_growth'] = construction.pct_change().rolling(self.spatial_window).mean()

        # 植被健康度（NDVI）
        if 'ndvi_index' in satellite_data.columns:
            ndvi = satellite_data['ndvi_index']
            metrics['vegetation_health'] = ndvi.rolling(self.spatial_window).mean()
            metrics['vegetation_change'] = ndvi.diff().rolling(self.spatial_window).mean()

        # 交通流量（基于车辆检测）
        if 'traffic_density' in satellite_data.columns:
            traffic = satellite_data['traffic_density']
            metrics['traffic_density_ma'] = traffic.rolling(self.spatial_window).mean()
            metrics['traffic_congestion'] = (traffic > traffic.quantile(0.8)).rolling(self.spatial_window).mean()

        # 工业活动
        if 'industrial_activity' in satellite_data.columns:
            industrial = satellite_data['industrial_activity']
            metrics['industrial_activity_ma'] = industrial.rolling(self.spatial_window).mean()
            metrics['industrial_heat_map'] = industrial.rolling(self.spatial_window).apply(
                lambda x: np.sum(x > np.percentile(x, 75))
            )

        # 港口活动
        if 'port_activity' in satellite_data.columns:
            port_activity = satellite_data['port_activity']
            metrics['port_utilization'] = port_activity.rolling(self.spatial_window).mean()
            metrics['shipping_traffic'] = port_activity.rolling(self.spatial_window).std()

        return metrics

    def calculate_foot_traffic_metrics(self, traffic_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算人流指标

        Args:
            traffic_data: 人流数据

        Returns:
            人流指标字典
        """
        metrics = {}

        # 总人流量
        if 'total_foot_traffic' in traffic_data.columns:
            total_traffic = traffic_data['total_foot_traffic']
            metrics['total_traffic_ma'] = total_traffic.rolling(self.spatial_window).mean()
            metrics['traffic_growth'] = total_traffic.pct_change().rolling(self.spatial_window).mean()

            # 人流量排名（百分位）
            metrics['traffic_percentile'] = total_traffic.rolling(self.spatial_window).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
            )

        # 零售人流
        if 'retail_traffic' in traffic_data.columns:
            retail_traffic = traffic_data['retail_traffic']
            metrics['retail_traffic_ma'] = retail_traffic.rolling(self.spatial_window).mean()
            metrics['retail_traffic_ratio'] = retail_traffic / (total_traffic + 1e-8)

        # 办公区人流
        if 'office_traffic' in traffic_data.columns:
            office_traffic = traffic_data['office_traffic']
            metrics['office_traffic_ma'] = office_traffic.rolling(self.spatial_window).mean()
            metrics['office_utilization'] = office_traffic.rolling(self.spatial_window).apply(
                lambda x: x.iloc[-1] / x.mean() if x.mean() > 0 else 1
            )

        # 娱乐场所人流
        if 'entertainment_traffic' in traffic_data.columns:
            entertainment_traffic = traffic_data['entertainment_traffic']
            metrics['entertainment_traffic_ma'] = entertainment_traffic.rolling(self.spatial_window).mean()
            metrics['weekend_vs_weekday'] = entertainment_traffic.rolling(7).mean() / \
                                             entertainment_traffic.rolling(7).mean().shift(5)

        # 交通枢纽人流
        if 'transit_hub_traffic' in traffic_data.columns:
            transit_traffic = traffic_data['transit_hub_traffic']
            metrics['transit_hub_utilization'] = transit_traffic.rolling(self.spatial_window).mean()
            metrics['transit_efficiency'] = transit_traffic.rolling(self.spatial_window).std() / \
                                          (transit_traffic.rolling(self.spatial_window).mean() + 1e-8)

        # 人流分布熵（衡量分散度）
        if 'traffic_distribution' in traffic_data.columns:
            traffic_dist = traffic_data['traffic_distribution']
            metrics['traffic_entropy'] = traffic_dist.rolling(self.spatial_window).apply(
                self._calculate_entropy_from_distribution
            )

        return metrics

    def _calculate_entropy_from_distribution(self, distribution_data: pd.Series) -> float:
        """从分布数据计算熵"""
        try:
            if isinstance(distribution_data.iloc[-1], dict):
                values = list(distribution_data.iloc[-1].values())
            elif isinstance(distribution_data.iloc[-1], (list, np.ndarray)):
                values = distribution_data.iloc[-1]
            else:
                values = [distribution_data.iloc[-1]]

            if len(values) == 0:
                return 0

            # 归一化
            values = np.array(values)
            values = values / (values.sum() + 1e-8)

            # 计算熵
            entropy = -np.sum(values * np.log(values + 1e-8))
            return entropy / np.log(len(values)) if len(values) > 1 else 0

        except:
            return 0

    def calculate_location_intelligence_metrics(self, location_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算位置智能指标

        Args:
            location_data: 位置数据

        Returns:
            位置智能指标字典
        """
        metrics = {}

        # 商店密度
        if 'store_density' in location_data.columns:
            store_density = location_data['store_density']
            metrics['retail_density'] = store_density.rolling(self.spatial_window).mean()
            metrics['retail_saturation'] = (store_density > store_density.quantile(0.8)).rolling(self.spatial_window).mean()

        # 餐厅密度
        if 'restaurant_density' in location_data.columns:
            restaurant_density = location_data['restaurant_density']
            metrics['dining_activity'] = restaurant_density.rolling(self.spatial_window).mean()
            metrics['food_beverage_index'] = restaurant_density.rolling(self.spatial_window).apply(
                lambda x: np.sum(x > np.percentile(x, 70))
            )

        # 住宅区活跃度
        if 'residential_activity' in location_data.columns:
            residential_activity = location_data['residential_activity']
            metrics['residential_vitality'] = residential_activity.rolling(self.spatial_window).mean()
            metrics['community_engagement'] = residential_activity.rolling(self.spatial_window).std()

        # 商业地产空置率
        if 'commercial_vacancy' in location_data.columns:
            vacancy_rate = location_data['commercial_vacancy']
            metrics['vacancy_rate_ma'] = vacancy_rate.rolling(self.spatial_window).mean()
            metrics['occupancy_trend'] = (1 - vacancy_rate).rolling(self.spatial_window).mean()

        # 新开业企业
        if 'new_business_openings' in location_data.columns:
            new_businesses = location_data['new_business_openings']
            metrics['business_growth_rate'] = new_businesses.rolling(self.spatial_window).mean()
            metrics['entrepreneurial_activity'] = new_businesses.rolling(self.spatial_window).apply(
                lambda x: np.sum(x > np.percentile(x, 60))
            )

        # 地点评分变化
        if 'location_ratings' in location_data.columns:
            ratings = location_data['location_ratings']
            metrics['location_quality_trend'] = ratings.rolling(self.spatial_window).mean()
            metrics['customer_satisfaction'] = (ratings > 4.0).rolling(self.spatial_window).mean()

        return metrics

    def calculate_weather_climate_metrics(self, weather_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算天气气候指标

        Args:
            weather_data: 天气数据

        Returns:
            天气气候指标字典
        """
        metrics = {}

        # 温度
        if 'temperature' in weather_data.columns:
            temperature = weather_data['temperature']
            metrics['temperature_ma'] = temperature.rolling(self.spatial_window).mean()
            metrics['temperature_anomaly'] = (temperature - temperature.rolling(30).mean()).rolling(self.spatial_window).mean()
            metrics['heating_degree_days'] = (18 - temperature).clip(lower=0).rolling(self.spatial_window).sum()
            metrics['cooling_degree_days'] = (temperature - 18).clip(lower=0).rolling(self.spatial_window).sum()

        # 降水量
        if 'precipitation' in weather_data.columns:
            precipitation = weather_data['precipitation']
            metrics['precipitation_ma'] = precipitation.rolling(self.spatial_window).mean()
            metrics['drought_index'] = (precipitation < precipitation.quantile(0.2)).rolling(self.spatial_window).mean()
            metrics['flood_risk'] = (precipitation > precipitation.quantile(0.9)).rolling(self.spatial_window).mean()

        # 风速
        if 'wind_speed' in weather_data.columns:
            wind_speed = weather_data['wind_speed']
            metrics['wind_energy_potential'] = wind_speed.rolling(self.spatial_window).apply(
                lambda x: np.mean(x**3)  # 风能与风速三次方成正比
            )
            metrics['storm_frequency'] = (wind_speed > wind_speed.quantile(0.95)).rolling(self.spatial_window).sum()

        # 日照时长
        if 'sunshine_hours' in weather_data.columns:
            sunshine = weather_data['sunshine_hours']
            metrics['solar_energy_potential'] = sunshine.rolling(self.spatial_window).mean()
            metrics['seasonal_variation'] = sunshine.rolling(self.spatial_window).std()

        # 湿度
        if 'humidity' in weather_data.columns:
            humidity = weather_data['humidity']
            metrics['humidity_discomfort'] = (humidity > 70).rolling(self.spatial_window).mean()
            metrics['agricultural_suitability'] = ((humidity > 40) & (humidity < 70)).rolling(self.spatial_window).mean()

        # 空气质量
        if 'air_quality_index' in weather_data.columns:
            aqi = weather_data['air_quality_index']
            metrics['air_quality_ma'] = aqi.rolling(self.spatial_window).mean()
            metrics['pollution_events'] = (aqi > 150).rolling(self.spatial_window).sum()

        return metrics

    def calculate_supply_chain_geospatial_metrics(self, supply_chain_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算供应链地理空间指标

        Args:
            supply_chain_data: 供应链数据

        Returns:
            供应链地理空间指标字典
        """
        metrics = {}

        # 港口活动
        if 'port_throughput' in supply_chain_data.columns:
            port_throughput = supply_chain_data['port_throughput']
            metrics['port_efficiency'] = port_throughput.rolling(self.spatial_window).mean()
            metrics['shipping_congestion'] = (port_throughput < port_throughput.quantile(0.3)).rolling(self.spatial_window).mean()

        # 物流中心活跃度
        if 'logistics_center_activity' in supply_chain_data.columns:
            logistics_activity = supply_chain_data['logistics_center_activity']
            metrics['logistics_utilization'] = logistics_activity.rolling(self.spatial_window).mean()
            metrics['distribution_efficiency'] = logistics_activity.rolling(self.spatial_window).std() / \
                                             (logistics_activity.rolling(self.spatial_window).mean() + 1e-8)

        # 运输路线密度
        if 'route_density' in supply_chain_data.columns:
            route_density = supply_chain_data['route_density']
            metrics['transportation_network_density'] = route_density.rolling(self.spatial_window).mean()
            metrics['connectivity_index'] = route_density.rolling(self.spatial_window).apply(
                lambda x: np.sum(x > np.percentile(x, 50)) / len(x)
            )

        # 库存分布
        if 'inventory_distribution' in supply_chain_data.columns:
            inventory_dist = supply_chain_data['inventory_distribution']
            metrics['inventory_concentration'] = inventory_dist.rolling(self.spatial_window).apply(
                lambda x: self._calculate_concentration_index(x)
            )
            metrics['supply_chain_resilience'] = 1 - metrics['inventory_concentration']

        # 交货时间
        if 'delivery_time' in supply_chain_data.columns:
            delivery_time = supply_chain_data['delivery_time']
            metrics['delivery_efficiency'] = (1 / (delivery_time + 1e-8)).rolling(self.spatial_window).mean()
            metrics['logistics_reliability'] = (delivery_time < delivery_time.quantile(0.7)).rolling(self.spatial_window).mean()

        return metrics

    def _calculate_concentration_index(self, distribution_data: pd.Series) -> float:
        """计算集中度指数（赫芬达尔指数）"""
        try:
            if isinstance(distribution_data.iloc[-1], dict):
                values = list(distribution_data.iloc[-1].values())
            elif isinstance(distribution_data.iloc[-1], (list, np.ndarray)):
                values = distribution_data.iloc[-1]
            else:
                values = [distribution_data.iloc[-1]]

            if len(values) == 0:
                return 0

            values = np.array(values)
            values = values / (values.sum() + 1e-8)  # 归一化

            hhi = np.sum(values**2)  # 赫芬达尔-赫希曼指数
            return hhi

        except:
            return 0

    def detect_geospatial_anomalies(self, satellite_data: pd.DataFrame,
                                   traffic_data: pd.DataFrame,
                                   location_data: pd.DataFrame) -> Dict[str, Any]:
        """
        检测地理空间异常

        Args:
            satellite_data: 卫星数据
            traffic_data: 人流数据
            location_data: 位置数据

        Returns:
            地理空间异常检测结果
        """
        anomalies = {}

        # 卫星图像异常
        if satellite_data is not None and len(satellite_data) > 0:
            satellite_anomalies = self._detect_satellite_anomalies(satellite_data)
            anomalies['satellite_anomalies'] = satellite_anomalies

        # 人流异常
        if traffic_data is not None and len(traffic_data) > 0:
            traffic_anomalies = self._detect_traffic_anomalies(traffic_data)
            anomalies['traffic_anomalies'] = traffic_anomalies

        # 位置异常
        if location_data is not None and len(location_data) > 0:
            location_anomalies = self._detect_location_anomalies(location_data)
            anomalies['location_anomalies'] = location_anomalies

        return anomalies

    def _detect_satellite_anomalies(self, satellite_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """检测卫星图像异常"""
        anomalies = []

        if 'night_light_intensity' in satellite_data.columns:
            night_light = satellite_data['night_light_intensity']

            # 检测灯光异常变化
            light_change = night_light.pct_change()
            extreme_changes = light_change[abs(light_change) > 0.5]  # 50%以上变化

            for date, change in extreme_changes.items():
                if pd.notna(change):
                    anomalies.append({
                        'date': date,
                        'type': 'night_light_change',
                        'severity': 'high' if abs(change) > 1.0 else 'medium',
                        'change_pct': change,
                        'description': f'夜间灯光变化{change:.1%}'
                    })

        return anomalies

    def _detect_traffic_anomalies(self, traffic_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """检测人流异常"""
        anomalies = []

        if 'total_foot_traffic' in traffic_data.columns:
            total_traffic = traffic_data['total_foot_traffic']

            # 检测人流量异常
            traffic_zscore = (total_traffic - total_traffic.rolling(30).mean()) / \
                           (total_traffic.rolling(30).std() + 1e-8)

            extreme_traffic = traffic_zscore[abs(traffic_zscore) > 2]  # 2个标准差

            for date, zscore in extreme_traffic.items():
                if pd.notna(zscore):
                    anomalies.append({
                        'date': date,
                        'type': 'traffic_anomaly',
                        'severity': 'high' if abs(zscore) > 3 else 'medium',
                        'zscore': zscore,
                        'description': f'人流量异常，Z-score: {zscore:.1f}'
                    })

        return anomalies

    def _detect_location_anomalies(self, location_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """检测位置异常"""
        anomalies = []

        if 'commercial_vacancy' in location_data.columns:
            vacancy_rate = location_data['commercial_vacancy']

            # 检测空置率异常
            vacancy_spike = vacancy_rate[vacancy_rate > vacancy_rate.quantile(0.95)]

            for date, rate in vacancy_spike.items():
                anomalies.append({
                    'date': date,
                    'type': 'vacancy_anomaly',
                    'severity': 'high' if rate > 0.3 else 'medium',
                    'vacancy_rate': rate,
                    'description': f'商业地产空置率异常: {rate:.1%}'
                })

        return anomalies

    def calculate_geospatial_economic_indicators(self, satellite_data: pd.DataFrame,
                                               traffic_data: pd.DataFrame,
                                               location_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算地理空间经济指标

        Args:
            satellite_data: 卫星数据
            traffic_data: 人流数据
            location_data: 位置数据

        Returns:
            地理空间经济指标DataFrame
        """
        # 计算各类指标
        satellite_metrics = self.calculate_satellite_imagery_metrics(satellite_data) if satellite_data is not None else {}
        traffic_metrics = self.calculate_foot_traffic_metrics(traffic_data) if traffic_data is not None else {}
        location_metrics = self.calculate_location_intelligence_metrics(location_data) if location_data is not None else {}

        # 创建指标DataFrame
        indicators = pd.DataFrame()

        # 添加卫星指标
        for name, series in satellite_metrics.items():
            indicators[f'satellite_{name}'] = series.iloc[-1] if len(series) > 0 else 0

        # 添加人流指标
        for name, series in traffic_metrics.items():
            indicators[f'traffic_{name}'] = series.iloc[-1] if len(series) > 0 else 0

        # 添加位置指标
        for name, series in location_metrics.items():
            indicators[f'location_{name}'] = series.iloc[-1] if len(series) > 0 else 0

        # 计算综合经济活跃度指数
        economic_activity_score = self._calculate_economic_activity_score(
            satellite_metrics, traffic_metrics, location_metrics
        )
        indicators['economic_activity_score'] = economic_activity_score

        # 检测异常
        anomalies = self.detect_geospatial_anomalies(satellite_data, traffic_data, location_data)

        # 异常计数
        total_anomalies = sum(len(anomaly_list) for anomaly_list in anomalies.values())
        indicators['anomaly_count'] = total_anomalies
        indicators['anomaly_severity'] = min(3, total_anomalies / 5)  # 每5个异常为1个严重级别

        # 发展势头指标
        momentum_score = self._calculate_development_momentum(
            satellite_metrics, traffic_metrics, location_metrics
        )
        indicators['development_momentum'] = momentum_score

        # 区域韧性指标
        resilience_score = self._calculate_regional_resilience(
            satellite_metrics, traffic_metrics, location_metrics
        )
        indicators['regional_resilience'] = resilience_score

        return indicators.T

    def _calculate_economic_activity_score(self, satellite_metrics: Dict[str, pd.Series],
                                        traffic_metrics: Dict[str, pd.Series],
                                        location_metrics: Dict[str, pd.Series]) -> float:
        """计算经济活跃度评分"""
        score_components = []

        # 卫星指标贡献
        if 'night_light_ma' in satellite_metrics:
            light_score = min(1, satellite_metrics['night_light_ma'].iloc[-1] / 100) if len(satellite_metrics['night_light_ma']) > 0 else 0
            score_components.append(light_score * 0.3)

        if 'construction_intensity' in satellite_metrics:
            construction_score = min(1, satellite_metrics['construction_intensity'].iloc[-1] / 50) if len(satellite_metrics['construction_intensity']) > 0 else 0
            score_components.append(construction_score * 0.2)

        # 人流指标贡献
        if 'total_traffic_ma' in traffic_metrics:
            traffic_score = min(1, traffic_metrics['total_traffic_ma'].iloc[-1] / 10000) if len(traffic_metrics['total_traffic_ma']) > 0 else 0
            score_components.append(traffic_score * 0.3)

        # 位置指标贡献
        if 'business_growth_rate' in location_metrics:
            business_score = min(1, location_metrics['business_growth_rate'].iloc[-1] / 100) if len(location_metrics['business_growth_rate']) > 0 else 0
            score_components.append(business_score * 0.2)

        if 'occupancy_trend' in location_metrics:
            occupancy_score = location_metrics['occupancy_trend'].iloc[-1] if len(location_metrics['occupancy_trend']) > 0 else 0
            score_components.append(occupancy_score * 0.2)

        # 综合评分
        if score_components:
            return np.mean(score_components)
        else:
            return 0.5

    def _calculate_development_momentum(self, satellite_metrics: Dict[str, pd.Series],
                                    traffic_metrics: Dict[str, pd.Series],
                                    location_metrics: Dict[str, pd.Series]) -> float:
        """计算发展势头"""
        momentum_indicators = []

        # 卫星势头
        if 'night_light_growth' in satellite_metrics:
            light_growth = satellite_metrics['night_light_growth'].iloc[-1] if len(satellite_metrics['night_light_growth']) > 0 else 0
            momentum_indicators.append(np.clip(light_growth * 10, -1, 1))  # 缩放并限制范围

        if 'construction_growth' in satellite_metrics:
            construction_growth = satellite_metrics['construction_growth'].iloc[-1] if len(satellite_metrics['construction_growth']) > 0 else 0
            momentum_indicators.append(np.clip(construction_growth * 5, -1, 1))

        # 人流势头
        if 'traffic_growth' in traffic_metrics:
            traffic_growth = traffic_metrics['traffic_growth'].iloc[-1] if len(traffic_metrics['traffic_growth']) > 0 else 0
            momentum_indicators.append(np.clip(traffic_growth * 3, -1, 1))

        # 综合势头
        if momentum_indicators:
            return np.mean(momentum_indicators)
        else:
            return 0

    def _calculate_regional_resilience(self, satellite_metrics: Dict[str, pd.Series],
                                    traffic_metrics: Dict[str, pd.Series],
                                    location_metrics: Dict[str, pd.Series]) -> float:
        """计算区域韧性"""
        resilience_factors = []

        # 多样性指标
        if 'traffic_entropy' in traffic_metrics:
            entropy = traffic_metrics['traffic_entropy'].iloc[-1] if len(traffic_metrics['traffic_entropy']) > 0 else 0
            resilience_factors.append(entropy)  # 熵越高表示分布越分散，韧性越强

        # 活性平衡
        if 'retail_traffic_ratio' in traffic_metrics and 'office_traffic_ma' in traffic_metrics:
            retail_balance = traffic_metrics['retail_traffic_ratio'].iloc[-1] if len(traffic_metrics['retail_traffic_ratio']) > 0 else 0.5
            office_activity = traffic_metrics['office_traffic_ma'].iloc[-1] if len(traffic_metrics['office_traffic_ma']) > 0 else 0.5
            balance_score = 1 - abs(retail_balance - 0.5) * 2  # 平衡得分
            resilience_factors.append(balance_score)

        # 稳定性指标
        if 'location_quality_trend' in location_metrics:
            quality_stability = location_metrics['location_quality_trend'].iloc[-1] if len(location_metrics['location_quality_trend']) > 0 else 0.5
            resilience_factors.append(quality_stability)

        if resilience_factors:
            return np.mean(resilience_factors)
        else:
            return 0.5


class UrbanDevelopmentTracker:
    """
    城市发展追踪器

    专门追踪城市发展和城市化进程
    """

    def __init__(self):
        """初始化城市发展追踪器"""
        self.development_stages = {
            'rural': {'population_density': 0, 'night_lights': 0, 'urban_infrastructure': 0},
            'suburban': {'population_density': 0.5, 'night_lights': 0.4, 'urban_infrastructure': 0.6},
            'urban': {'population_density': 0.8, 'night_lights': 0.7, 'urban_infrastructure': 0.8},
            'metropolitan': {'population_density': 1.0, 'night_lights': 1.0, 'urban_infrastructure': 1.0}
        }

    def classify_development_stage(self, geospatial_indicators: pd.DataFrame) -> Dict[str, Any]:
        """
        分类城市发展阶段

        Args:
            geospatial_indicators: 地理空间指标

        Returns:
            发展阶段分类结果
        """
        if geospatial_indicators.empty:
            return {'stage': 'unknown', 'confidence': 0}

        # 提取关键指标
        indicators_dict = geospatial_indicators.to_dict()

        # 计算发展得分
        development_score = 0
        score_components = 0

        # 人口密度得分
        if 'satellite_night_light_ma' in indicators_dict:
            light_score = min(1, indicators_dict['satellite_night_light_ma'][0] / 100)
            development_score += light_score * 0.3
            score_components += 1

        if 'traffic_total_traffic_ma' in indicators_dict:
            traffic_score = min(1, indicators_dict['traffic_total_traffic_ma'][0] / 10000)
            development_score += traffic_score * 0.3
            score_components += 1

        if 'location_business_growth_rate' in indicators_dict:
            business_score = min(1, indicators_dict['location_business_growth_rate'][0] / 100)
            development_score += business_score * 0.2
            score_components += 1

        if 'location_occupancy_trend' in indicators_dict:
            occupancy_score = indicators_dict['location_occupancy_trend'][0]
            development_score += occupancy_score * 0.2
            score_components += 1

        # 平均得分
        if score_components > 0:
            development_score /= score_components
        else:
            development_score = 0.5

        # 分类发展阶段
        if development_score < 0.25:
            stage = 'rural'
        elif development_score < 0.5:
            stage = 'suburban'
        elif development_score < 0.75:
            stage = 'urban'
        else:
            stage = 'metropolitan'

        # 计算置信度
        if stage == 'rural':
            confidence = 1 - development_score * 2
        elif stage == 'metropolitan':
            confidence = (development_score - 0.75) * 4
        else:
            confidence = 0.8

        return {
            'stage': stage,
            'development_score': development_score,
            'confidence': confidence,
            'next_stage_transition': self._estimate_transition_time(development_score, stage)
        }

    def _estimate_transition_time(self, current_score: float, current_stage: str) -> float:
        """估计发展阶段转换时间"""
        stage_thresholds = {'rural': 0.25, 'suburban': 0.5, 'urban': 0.75, 'metropolitan': 1.0}

        if current_stage in stage_thresholds:
            next_threshold = stage_thresholds.get(list(stage_thresholds.keys())[
                list(stage_thresholds.keys()).index(current_stage) + 1], 1.0)

            if current_score < next_threshold:
                # 假设每年发展5%
                years_to_transition = (next_threshold - current_score) / 0.05
                return max(0, years_to_transition)

        return float('inf')


def create_geospatial_features(satellite_data: pd.DataFrame = None,
                             traffic_data: pd.DataFrame = None,
                             location_data: pd.DataFrame = None,
                             weather_data: pd.DataFrame = None,
                             supply_chain_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    创建地理空间特征

    Args:
        satellite_data: 卫星数据
        traffic_data: 人流数据
        location_data: 位置数据
        weather_data: 天气数据
        supply_chain_data: 供应链数据

    Returns:
        地理空间特征DataFrame
    """
    analyzer = GeospatialAnalyzer()
    indicators = analyzer.calculate_geospatial_economic_indicators(
        satellite_data, traffic_data, location_data
    )
    return indicators


# 主要功能函数
def calculate_geospatial_indicators(satellite_data: pd.DataFrame = None,
                                 traffic_data: pd.DataFrame = None,
                                 location_data: pd.DataFrame = None,
                                 weather_data: pd.DataFrame = None,
                                 supply_chain_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    计算所有地理空间指标

    Args:
        satellite_data: 卫星数据DataFrame（可选）
        traffic_data: 人流数据DataFrame（可选）
        location_data: 位置数据DataFrame（可选）
        weather_data: 天气数据DataFrame（可选）
        supply_chain_data: 供应链数据DataFrame（可选）

    Returns:
        包含所有指标值的DataFrame
    """
    return create_geospatial_features(satellite_data, traffic_data, location_data, weather_data, supply_chain_data)


# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # 模拟卫星数据
    satellite_data = pd.DataFrame({
        'night_light_intensity': np.random.uniform(50, 200, 100),
        'construction_activity': np.random.uniform(10, 50, 100),
        'ndvi_index': np.random.uniform(0.3, 0.8, 100),
        'traffic_density': np.random.uniform(20, 80, 100),
        'industrial_activity': np.random.uniform(30, 90, 100),
        'port_activity': np.random.uniform(40, 100, 100)
    }, index=dates)

    # 模拟人流数据
    traffic_data = pd.DataFrame({
        'total_foot_traffic': np.random.randint(5000, 20000, 100),
        'retail_traffic': np.random.randint(2000, 8000, 100),
        'office_traffic': np.random.randint(3000, 12000, 100),
        'entertainment_traffic': np.random.randint(1000, 5000, 100),
        'transit_hub_traffic': np.random.randint(4000, 15000, 100),
        'traffic_distribution': [{'area1': 1000, 'area2': 1500, 'area3': 800} for _ in range(100)]
    }, index=dates)

    # 模拟位置数据
    location_data = pd.DataFrame({
        'store_density': np.random.uniform(5, 25, 100),
        'restaurant_density': np.random.uniform(3, 15, 100),
        'residential_activity': np.random.uniform(50, 90, 100),
        'commercial_vacancy': np.random.uniform(0.05, 0.15, 100),
        'new_business_openings': np.random.randint(1, 20, 100),
        'location_ratings': np.random.uniform(3.5, 4.8, 100)
    }, index=dates)

    # 模拟天气数据
    weather_data = pd.DataFrame({
        'temperature': np.random.uniform(10, 30, 100),
        'precipitation': np.random.uniform(0, 50, 100),
        'wind_speed': np.random.uniform(5, 25, 100),
        'sunshine_hours': np.random.uniform(4, 12, 100),
        'humidity': np.random.uniform(40, 80, 100),
        'air_quality_index': np.random.uniform(30, 150, 100)
    }, index=dates)

    # 模拟供应链数据
    supply_chain_data = pd.DataFrame({
        'port_throughput': np.random.uniform(1000, 5000, 100),
        'logistics_center_activity': np.random.uniform(500, 2500, 100),
        'route_density': np.random.uniform(20, 100, 100),
        'inventory_distribution': [{'warehouse1': 1000, 'warehouse2': 1500, 'warehouse3': 800} for _ in range(100)],
        'delivery_time': np.random.uniform(1, 7, 100)
    }, index=dates)

    # 计算指标
    try:
        indicators = calculate_geospatial_indicators(
            satellite_data, traffic_data, location_data, weather_data, supply_chain_data
        )
        print("地理空间指标计算成功!")
        print(f"指标数量: {indicators.shape[0]}")
        print("最新指标值:")
        print(indicators)

        # 城市发展阶段分类
        tracker = UrbanDevelopmentTracker()
        development_stage = tracker.classify_development_stage(indicators)
        print(f"\n城市发展阶段: {development_stage['stage']}")
        print(f"发展得分: {development_stage['development_score']:.2f}")
        print(f"置信度: {development_stage['confidence']:.2f}")

    except Exception as e:
        print(f"计算错误: {e}")