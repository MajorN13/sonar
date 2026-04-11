"""
Содержание:
1. Фильтр Калмана (EKF) для одного датчика
2. Связанный фильтр для всей сети датчиков
3. Анализ ковариации и доверительных интервалов

Фильтр Калмана объединяет:
- Модель движения (предсказание): x_{k+1} = F * x_k + w_k
- Модель измерений (коррекция): z_k = H * x_k + v_k

Где:
- x_k — состояние (позиция, скорость)
- z_k — измерения (расстояния от эхолокации)
- F — матрица перехода состояния
- H — матрица наблюдения
- w_k ~ N(0, Q) — шум процесса
- v_k ~ N(0, R) — шум измерений
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from Simulation1 import SystemParams, PhysicalModel, EcholocationModel, Triangulation


# ФИЛЬТР КАЛМАНА ДЛЯ ОДНОГО ДАТЧИКА

@dataclass
class KalmanState:
    """Состояние фильтра Калмана для одного датчика."""

    # Вектор состояния: [x, y, vx, vy]
    x: np.ndarray  # shape (4,)

    # Ковариационная матрица состояния
    P: np.ndarray  # shape (4, 4)

    def position(self) -> np.ndarray:
        """Получить позицию (x, y)."""
        return self.x[:2]

    def velocity(self) -> np.ndarray:
        """Получить скорость (vx, vy)."""
        return self.x[2:]

    def position_covariance(self) -> np.ndarray:
        """Получить ковариацию позиции (2x2)."""
        return self.P[:2, :2]


class SingleSensorKalman:
    """
    Фильтр Калмана для одного датчика.

    Состояние: x = [x, y, vx, vy]^T

    Модель движения (постоянная скорость с ускорением как шумом):
        x_{k+1} = F * x_k + w_k

        F = [[1, 0, dt, 0 ],
             [0, 1, 0,  dt],
             [0, 0, 1,  0 ],
             [0, 0, 0,  1 ]]

    Модель измерений — нелинейная (расстояния до соседей),
    поэтому используем расширенный фильтр Калмана (EKF).
    """

    def __init__(self,
                 dt: float = 0.1,
                 process_noise: float = 1.0,
                 measurement_noise: float = 0.2):
        """
        Args:
            dt: Шаг по времени
            process_noise: Стандартное отклонение ускорения (м/с²)
            measurement_noise: Стандартное отклонение измерений расстояния (м)
        """
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # Матрица перехода состояния F
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Матрица шума процесса Q
        q = process_noise ** 2
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4

        self.Q = q * np.array([
            [dt4 / 4, 0, dt3 / 2, 0],
            [0, dt4 / 4, 0, dt3 / 2],
            [dt3 / 2, 0, dt2, 0],
            [0, dt3 / 2, 0, dt2]
        ])

        # Шум измерений
        self.R_single = measurement_noise ** 2

    def init_state(self, position: np.ndarray,
                   velocity: Optional[np.ndarray] = None,
                   position_uncertainty: float = 1.0,
                   velocity_uncertainty: float = 2.0) -> KalmanState:
        """
        Инициализировать состояние фильтра.

        Args:
            position: Начальная позиция (x, y)
            velocity: Начальная скорость (vx, vy), по умолчанию (0, 0)
            position_uncertainty: Начальная неопределённость позиции
            velocity_uncertainty: Начальная неопределённость скорости
        """
        if velocity is None:
            velocity = np.zeros(2)

        x = np.concatenate([position, velocity])

        P = np.diag([
            position_uncertainty ** 2,
            position_uncertainty ** 2,
            velocity_uncertainty ** 2,
            velocity_uncertainty ** 2
        ])

        return KalmanState(x=x, P=P)

    def predict(self, state: KalmanState) -> KalmanState:
        """
        Шаг предсказания (prediction step).

        x_{k|k-1} = F * x_{k-1|k-1}
        P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        """
        x_pred = self.F @ state.x
        P_pred = self.F @ state.P @ self.F.T + self.Q

        return KalmanState(x=x_pred, P=P_pred)

    def update_with_distance(self, state: KalmanState,
                             anchor_pos: np.ndarray,
                             measured_distance: float) -> KalmanState:
        """
        Шаг коррекции по измерению расстояния до известной точки.

        Модель измерения (нелинейная):
            z = ||position - anchor|| + v

        Используем EKF: линеаризуем вокруг текущей оценки.

        Args:
            state: Текущее состояние
            anchor_pos: Позиция известной точки (якорь)
            measured_distance: Измеренное расстояние
        """
        pos = state.x[:2]

        # Предсказанное расстояние
        delta = pos - anchor_pos
        predicted_distance = np.linalg.norm(delta)

        if predicted_distance < 1e-10:
            # Избегаем деления на ноль
            return state

        # Якобиан измерения H (размер 1x4)
        # d(distance)/d(x) = (x - ax) / distance
        # d(distance)/d(y) = (y - ay) / distance
        # d(distance)/d(vx) = 0
        # d(distance)/d(vy) = 0
        H = np.zeros((1, 4))
        H[0, 0] = delta[0] / predicted_distance
        H[0, 1] = delta[1] / predicted_distance

        # Инновация (разница между измерением и предсказанием)
        y = measured_distance - predicted_distance

        # Ковариация инновации
        R = np.array([[self.R_single]])
        S = H @ state.P @ H.T + R

        # Коэффициент Калмана
        K = state.P @ H.T @ np.linalg.inv(S)

        # Обновление состояния
        x_new = state.x + K.flatten() * y

        # Обновление ковариации (формула Джозефа для численной стабильности)
        I = np.eye(4)
        IKH = I - K @ H
        P_new = IKH @ state.P @ IKH.T + K @ R @ K.T

        return KalmanState(x=x_new, P=P_new)

    def update_with_multiple_distances(self, state: KalmanState,
                                       anchors: List[np.ndarray],
                                       distances: List[float]) -> KalmanState:
        """
        Шаг коррекции по нескольким измерениям расстояний.

        Args:
            state: Текущее состояние
            anchors: Список позиций известных точек
            distances: Список измеренных расстояний
        """
        # Применяем измерения последовательно
        current_state = state
        for anchor, dist in zip(anchors, distances):
            current_state = self.update_with_distance(current_state, anchor, dist)

        return current_state


# СВЯЗАННЫЙ ФИЛЬТР ДЛЯ ВСЕЙ СЕТИ ДАТЧИКОВ

class NetworkKalmanFilter:
    """
    Фильтр Калмана для всей сети датчиков.

    Учитывает:
    1. Связи между датчиками через измерения эхолокации
    2. Известные позиции точек крепления (якоря)
    3. Корреляции между соседними датчиками

    Алгоритм:
    1. Предсказание для всех датчиков
    2. Итеративная коррекция:
       - Сначала по связям с якорями (точки крепления)
       - Затем по связям между датчиками (от известных к неизвестным)
    """

    def __init__(self, params: SystemParams,
                 dt: float = 0.1,
                 process_noise: float = 1.0,
                 measurement_noise: float = 0.2):
        """
        Args:
            params: Параметры системы
            dt: Шаг по времени
            process_noise: Шум процесса (ускорение)
            measurement_noise: Шум измерений
        """
        self.params = params
        self.dt = dt

        # Создаём фильтры для каждого датчика
        self.filters: List[List[SingleSensorKalman]] = []
        self.states: List[List[KalmanState]] = []

        for i in range(params.N):
            row_filters = []
            row_states = []
            for j in range(params.M):
                kf = SingleSensorKalman(dt, process_noise, measurement_noise)
                row_filters.append(kf)
                row_states.append(None)  # Будет инициализировано позже
            self.filters.append(row_filters)
            self.states.append(row_states)

    def get_anchor(self, rope_idx: int) -> np.ndarray:
        """Получить позицию точки крепления."""
        p = self.params
        x = (rope_idx - (p.N - 1) / 2) * p.d
        y = 20.0
        return np.array([x, y])

    def initialize(self, initial_positions: np.ndarray,
                   initial_velocities: Optional[np.ndarray] = None):
        """
        Инициализировать все фильтры.

        Args:
            initial_positions: Начальные позиции, shape (N, M, 2)
            initial_velocities: Начальные скорости, shape (N, M, 2), опционально
        """
        p = self.params

        for i in range(p.N):
            for j in range(p.M):
                pos = initial_positions[i, j]
                vel = initial_velocities[i, j] if initial_velocities is not None else None

                self.states[i][j] = self.filters[i][j].init_state(
                    position=pos,
                    velocity=vel,
                    position_uncertainty=0.5,
                    velocity_uncertainty=1.0
                )

    def predict_all(self):
        """Шаг предсказания для всех датчиков."""
        p = self.params

        for i in range(p.N):
            for j in range(p.M):
                self.states[i][j] = self.filters[i][j].predict(self.states[i][j])

    def update_all(self, measurements: dict):
        """
        Шаг коррекции для всех датчиков.

        Алгоритм итеративной коррекции:
        1. Обрабатываем датчики строка за строкой (j = 0, 1, ..., M-1)
        2. В каждой строке слева направо (i = 0, 1, ..., N-1)
        3. Для каждого датчика используем:
           - Расстояние до верхнего соседа (j-1) или якоря (j=0)
           - Расстояние до левого соседа (i-1), если есть

        Args:
            measurements: Измерения от EcholocationModel
        """
        p = self.params
        vertical = measurements['vertical']
        horizontal = measurements['horizontal']

        # Проходим строка за строкой
        for j in range(p.M):
            for i in range(p.N):
                anchors = []
                distances = []

                # 1. Связь с верхним соседом (или якорем)
                if j == 0:
                    # Первый ряд — связь с точкой крепления (известна точно)
                    anchor = self.get_anchor(i)
                    anchors.append(anchor)
                    distances.append(vertical[i, j])
                else:
                    # Связь с датчиком выше (уже обновлён)
                    upper_pos = self.states[i][j - 1].position()
                    anchors.append(upper_pos)
                    distances.append(vertical[i, j])

                # 2. Связь с левым соседом (если есть и уже обновлён)
                if i > 0:
                    left_pos = self.states[i - 1][j].position()
                    anchors.append(left_pos)
                    distances.append(horizontal[i - 1, j])

                # Применяем коррекцию
                self.states[i][j] = self.filters[i][j].update_with_multiple_distances(
                    self.states[i][j], anchors, distances
                )

    def step(self, measurements: dict):
        """
        Полный шаг фильтра: предсказание + коррекция.

        Args:
            measurements: Измерения эхолокации
        """
        self.predict_all()
        self.update_all(measurements)

    def get_positions(self) -> np.ndarray:
        """Получить текущие оценки позиций всех датчиков."""
        p = self.params
        positions = np.zeros((p.N, p.M, 2))

        for i in range(p.N):
            for j in range(p.M):
                positions[i, j] = self.states[i][j].position()

        return positions

    def get_velocities(self) -> np.ndarray:
        """Получить текущие оценки скоростей всех датчиков."""
        p = self.params
        velocities = np.zeros((p.N, p.M, 2))

        for i in range(p.N):
            for j in range(p.M):
                velocities[i, j] = self.states[i][j].velocity()

        return velocities

    def get_position_uncertainties(self) -> np.ndarray:
        """
        Получить неопределённости позиций (1-sigma) для всех датчиков.

        Returns:
            Массив shape (N, M) с величиной неопределённости (среднее по x и y)
        """
        p = self.params
        uncertainties = np.zeros((p.N, p.M))

        for i in range(p.N):
            for j in range(p.M):
                cov = self.states[i][j].position_covariance()
                # Берём среднее из дисперсий x и y
                uncertainties[i, j] = np.sqrt((cov[0, 0] + cov[1, 1]) / 2)

        return uncertainties

    def get_covariance_ellipse_params(self, i: int, j: int,
                                      n_std: float = 2.0) -> Tuple[float, float, float]:
        """
        Получить параметры эллипса неопределённости для датчика (i, j).

        Args:
            i, j: Индексы датчика
            n_std: Количество стандартных отклонений (2 = 95% доверительный интервал)

        Returns:
            (width, height, angle_degrees): Параметры эллипса
        """
        cov = self.states[i][j].position_covariance()

        # Собственные значения и векторы
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Сортируем по убыванию
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Угол главной оси
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        # Размеры эллипса
        width = 2 * n_std * np.sqrt(eigenvalues[0])
        height = 2 * n_std * np.sqrt(eigenvalues[1])

        return width, height, angle

# СРАВНЕНИЕ МЕТОДОВ И АНАЛИЗ

class FilterComparison:
    """Сравнение триангуляции с фильтром Калмана и без него."""

    def __init__(self, params: SystemParams):
        self.params = params
        self.model = PhysicalModel(params)
        self.echolocation = EcholocationModel(params, noise_std=0.2)
        self.triangulation = Triangulation(params)
        self.kalman = NetworkKalmanFilter(params, dt=0.1,
                                          process_noise=0.5,
                                          measurement_noise=0.2)

    def run_comparison(self, duration: float = 20.0,
                       dt: float = 0.1) -> Dict:
        """
        Запустить сравнение методов.

        Args:
            duration: Продолжительность симуляции
            dt: Шаг по времени

        Returns:
            Словарь с результатами
        """
        n_steps = int(duration / dt)

        # Хранилища результатов
        times = np.zeros(n_steps)
        true_positions = np.zeros((n_steps, self.params.N, self.params.M, 2))
        triang_positions = np.zeros_like(true_positions)
        kalman_positions = np.zeros_like(true_positions)
        kalman_uncertainties = np.zeros((n_steps, self.params.N, self.params.M))

        triang_errors = np.zeros(n_steps)
        kalman_errors = np.zeros(n_steps)

        # Инициализация
        self.model.reset()

        # Прогрев модели
        for _ in range(50):
            self.model.step(dt)

        # Первое измерение для инициализации Калмана
        initial_measurements = self.echolocation.measure_distances(
            self.model.positions, add_noise=True
        )
        initial_estimate = self.triangulation.triangulate_iterative(initial_measurements)
        self.kalman.initialize(initial_estimate)

        # Основной цикл
        for step in range(n_steps):
            times[step] = step * dt

            # Симуляция физики
            for _ in range(int(dt / 0.01)):
                self.model.step(0.01)

            # Истинные позиции
            true_pos = self.model.positions.copy()
            true_positions[step] = true_pos

            # Измерения
            measurements = self.echolocation.measure_distances(true_pos, add_noise=True)

            # Метод 1: Простая триангуляция
            triang_est = self.triangulation.triangulate_iterative(measurements)
            triang_positions[step] = triang_est

            # Метод 2: Фильтр Калмана
            self.kalman.step(measurements)
            kalman_est = self.kalman.get_positions()
            kalman_positions[step] = kalman_est
            kalman_uncertainties[step] = self.kalman.get_position_uncertainties()

            # Ошибки
            triang_errors[step] = np.sqrt(np.mean(
                np.sum((triang_est - true_pos) ** 2, axis=2)
            ))
            kalman_errors[step] = np.sqrt(np.mean(
                np.sum((kalman_est - true_pos) ** 2, axis=2)
            ))

        return {
            'times': times,
            'true_positions': true_positions,
            'triang_positions': triang_positions,
            'kalman_positions': kalman_positions,
            'kalman_uncertainties': kalman_uncertainties,
            'triang_errors': triang_errors,
            'kalman_errors': kalman_errors
        }

    def plot_error_comparison(self, results: Dict, save_path: Optional[str] = None):
        """Построить график сравнения ошибок."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        times = results['times']

        # График 1: Ошибки во времени
        ax1 = axes[0]
        ax1.plot(times, results['triang_errors'], 'r-', alpha=0.7,
                 label='Триангуляция', linewidth=1)
        ax1.plot(times, results['kalman_errors'], 'b-', alpha=0.7,
                 label='Фильтр Калмана', linewidth=1)
        ax1.set_ylabel('RMSE (м)')
        ax1.set_title('Сравнение ошибок позиционирования')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Средние значения
        mean_triang = np.mean(results['triang_errors'])
        mean_kalman = np.mean(results['kalman_errors'])
        ax1.axhline(mean_triang, color='r', linestyle='--', alpha=0.5)
        ax1.axhline(mean_kalman, color='b', linestyle='--', alpha=0.5)
        ax1.text(times[-1] * 0.95, mean_triang, f'{mean_triang:.3f}',
                 color='r', va='bottom', ha='right')
        ax1.text(times[-1] * 0.95, mean_kalman, f'{mean_kalman:.3f}',
                 color='b', va='top', ha='right')

        # График 2: Улучшение от Калмана
        ax2 = axes[1]
        improvement = (results['triang_errors'] - results['kalman_errors']) / results['triang_errors'] * 100
        ax2.fill_between(times, 0, improvement, alpha=0.3, color='green',
                         where=improvement > 0, label='Калман лучше')
        ax2.fill_between(times, 0, improvement, alpha=0.3, color='red',
                         where=improvement < 0, label='Триангуляция лучше')
        ax2.axhline(0, color='k', linewidth=0.5)
        ax2.set_xlabel('Время (с)')
        ax2.set_ylabel('Улучшение (%)')
        ax2.set_title('Относительное улучшение от фильтра Калмана')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        mean_improvement = np.mean(improvement)
        ax2.axhline(mean_improvement, color='green', linestyle='--', alpha=0.5)
        ax2.text(times[-1] * 0.95, mean_improvement, f'Среднее: {mean_improvement:.1f}%',
                 va='bottom', ha='right', color='green')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)

        return fig, axes

    def plot_trajectory_comparison(self, results: Dict,
                                   sensor_i: int = 2, sensor_j: int = 2,
                                   save_path: Optional[str] = None):
        """
        Построить траектории для одного датчика.

        Args:
            results: Результаты сравнения
            sensor_i, sensor_j: Индексы датчика
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        true_traj = results['true_positions'][:, sensor_i, sensor_j, :]
        triang_traj = results['triang_positions'][:, sensor_i, sensor_j, :]
        kalman_traj = results['kalman_positions'][:, sensor_i, sensor_j, :]

        # Истинная траектория
        ax.plot(true_traj[:, 0], true_traj[:, 1], 'g-', linewidth=2,
                label='Истинная', alpha=0.8)

        # Триангуляция
        ax.plot(triang_traj[:, 0], triang_traj[:, 1], 'r.', markersize=3,
                label='Триангуляция', alpha=0.5)

        # Калман
        ax.plot(kalman_traj[:, 0], kalman_traj[:, 1], 'b-', linewidth=1.5,
                label='Фильтр Калмана', alpha=0.8)

        # Начальная и конечная точки
        ax.scatter(true_traj[0, 0], true_traj[0, 1], s=100, c='green',
                   marker='o', zorder=5, label='Начало')
        ax.scatter(true_traj[-1, 0], true_traj[-1, 1], s=100, c='green',
                   marker='s', zorder=5, label='Конец')

        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_title(f'Траектория датчика ({sensor_i}, {sensor_j})')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)

        return fig, ax

    def plot_uncertainty_map(self, results: Dict, time_idx: int = -1,
                             save_path: Optional[str] = None):
        """
        Построить карту неопределённостей.

        Args:
            results: Результаты сравнения
            time_idx: Индекс временного шага (-1 = последний)
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        positions = results['kalman_positions'][time_idx]
        uncertainties = results['kalman_uncertainties'][time_idx]
        true_pos = results['true_positions'][time_idx]

        # Точки крепления
        anchors = np.array([
            [(i - (self.params.N - 1) / 2) * self.params.d, 20.0]
            for i in range(self.params.N)
        ])
        ax.scatter(anchors[:, 0], anchors[:, 1], s=100, c='blue',
                   marker='s', label='Точки крепления', zorder=5)

        # Датчики с эллипсами неопределённости
        for i in range(self.params.N):
            for j in range(self.params.M):
                pos = positions[i, j]
                true_p = true_pos[i, j]

                # Истинная позиция
                ax.scatter(true_p[0], true_p[1], s=40, c='green',
                           marker='o', alpha=0.7, zorder=4)

                # Оценённая позиция
                ax.scatter(pos[0], pos[1], s=30, c='red',
                           marker='x', zorder=6)

                # Эллипс неопределённости (2-sigma)
                width, height, angle = self.kalman.get_covariance_ellipse_params(i, j, n_std=2)
                ellipse = Ellipse(xy=pos, width=width, height=height, angle=angle,
                                  fill=False, color='blue', alpha=0.5, linewidth=1)
                ax.add_patch(ellipse)

        # Верёвки
        for i in range(self.params.N):
            rope_x = [anchors[i, 0]] + [positions[i, j, 0] for j in range(self.params.M)]
            rope_y = [anchors[i, 1]] + [positions[i, j, 1] for j in range(self.params.M)]
            ax.plot(rope_x, rope_y, 'k-', alpha=0.3, linewidth=1)

        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_title(f'Оценки с эллипсами неопределённости (2σ)\n'
                     f't = {results["times"][time_idx]:.1f} с')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Легенда
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='blue',
                   markersize=10, label='Точки крепления'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                   markersize=8, label='Истинные позиции'),
            Line2D([0], [0], marker='x', color='red', markersize=8,
                   label='Оценки Калмана'),
            Ellipse((0, 0), 1, 1, fill=False, color='blue', label='2σ эллипс')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)

        return fig, ax


# ДЕМОНСТРАЦИЯ
def run_kalman_demo():
    """Демонстрация фильтра Калмана."""

    print("=" * 60)
    print("ФИЛЬТР КАЛМАНА ДЛЯ СГЛАЖИВАНИЯ ТРАЕКТОРИЙ")
    print("=" * 60)

    # Параметры
    params = SystemParams(
        N=5, M=5,
        L=10.0, d=8.0,
        k=100.0, gamma=5.0,
        wave_amp=2.0
    )

    print(f"\nПараметры:")
    print(f"  Датчиков: {params.N} x {params.M} = {params.N * params.M}")
    print(f"  Шум измерений: σ = 0.2 м")
    print(f"  Шум процесса: σ = 0.5 м/с²")

    # Сравнение
    print("\n" + "-" * 40)
    print("Запуск сравнения методов...")

    comparison = FilterComparison(params)
    results = comparison.run_comparison(duration=20.0, dt=0.1)

    print(f"\nРезультаты (усреднённые по времени):")
    print(f"  Триангуляция RMSE:    {np.mean(results['triang_errors']):.4f} м")
    print(f"  Фильтр Калмана RMSE:  {np.mean(results['kalman_errors']):.4f} м")

    improvement = (1 - np.mean(results['kalman_errors']) / np.mean(results['triang_errors'])) * 100
    print(f"  Улучшение:            {improvement:.1f}%")

    # Графики
    print("\n" + "-" * 40)
    print("Создание визуализаций...")

    fig1, _ = comparison.plot_error_comparison(results, 'kalman_error_comparison.png')
    print("  Сохранено: kalman_error_comparison.png")

    fig2, _ = comparison.plot_trajectory_comparison(results, 2, 2, 'kalman_trajectory.png')
    print("  Сохранено: kalman_trajectory.png")

    fig3, _ = comparison.plot_uncertainty_map(results, -1, 'kalman_uncertainty.png')
    print("  Сохранено: kalman_uncertainty.png")

    print("\n" + "=" * 60)
    print("Демонстрация завершена!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_kalman_demo()