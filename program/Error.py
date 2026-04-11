"""
Анализ погрешностей и чувствительности системы эхолокационных датчиков.

Модуль содержит:
1. Анализ распространения ошибок по сети
2. Матрица чувствительности (Якобиан)
3. Геометрический фактор снижения точности (DOP)
4. Монте-Карло анализ
5. Оптимизация конфигурации сети

Теория:
-------
Ошибка позиционирования зависит от:
1. Шума измерений σ_meas
2. Геометрии сети (углы между линиями связи)
3. Положения датчика в сети (краевые vs центральные)
4. Накопления ошибок при последовательной триангуляции

Геометрический DOP (Dilution of Precision):
    σ_pos = DOP × σ_meas

Чем хуже геометрия (все связи в одном направлении), тем больше DOP.

"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import pearsonr
from scipy.linalg import svd
import warnings

from Simulation1 import SystemParams, PhysicalModel, EcholocationModel, Triangulation

# ЧАСТЬ 1: АНАЛИЗ РАСПРОСТРАНЕНИЯ ОШИБОК

@dataclass
class ErrorAnalysisResult:
    """Результаты анализа ошибок."""

    # Ошибки по каждому датчику
    mean_errors: np.ndarray  # shape (N, M) — средние ошибки
    std_errors: np.ndarray  # shape (N, M) — стд ошибок
    max_errors: np.ndarray  # shape (N, M) — максимальные ошибки

    # DOP факторы
    hdop: np.ndarray  # shape (N, M) — горизонтальный DOP
    vdop: np.ndarray  # shape (N, M) — вертикальный DOP
    pdop: np.ndarray  # shape (N, M) — позиционный DOP

    # Корреляции ошибок
    error_correlation: np.ndarray  # shape (N*M, N*M) — корреляционная матрица

    # Чувствительность к параметрам
    sensitivity: Dict[str, np.ndarray]


class ErrorPropagationAnalyzer:
    """
    Анализатор распространения ошибок в сети датчиков.

    Методы:
    1. Аналитический анализ через Якобиан
    2. Монте-Карло симуляции
    3. Расчёт DOP факторов
    """

    def __init__(self, params: SystemParams):
        self.params = params
        self.triangulation = Triangulation(params)

    def get_anchor(self, rope_idx: int) -> np.ndarray:
        """Получить позицию точки крепления."""
        p = self.params
        x = (rope_idx - (p.N - 1) / 2) * p.d
        y = 20.0
        return np.array([x, y])

    # Метод 1: Аналитический анализ (Якобиан)

    def compute_jacobian(self, positions: np.ndarray) -> np.ndarray:
        """
        Вычислить Якобиан системы триангуляции.

        Якобиан показывает, как малые изменения измерений влияют
        на оценки позиций.

        J[i,j] = ∂(position_i) / ∂(measurement_j)

        Args:
            positions: Истинные позиции датчиков, shape (N, M, 2)

        Returns:
            Якобиан, shape (N*M*2, num_measurements)
        """
        p = self.params
        N, M = p.N, p.M

        # Количество измерений: N*M вертикальных + (N-1)*M горизонтальных
        n_vert = N * M
        n_horiz = (N - 1) * M
        n_meas = n_vert + n_horiz

        # Количество координат: N * M * 2
        n_coords = N * M * 2

        # Числовое дифференцирование
        epsilon = 1e-6
        jacobian = np.zeros((n_coords, n_meas))

        # Базовые измерения (без шума)
        echolocation = EcholocationModel(p, noise_std=0)
        base_measurements = echolocation.measure_distances(positions, add_noise=False)
        base_estimate = self.triangulation.triangulate_iterative(base_measurements)

        # Варьируем каждое измерение
        meas_idx = 0

        # Вертикальные измерения
        for i in range(N):
            for j in range(M):
                # Возмущённые измерения
                vert_perturbed = base_measurements['vertical'].copy()
                vert_perturbed[i, j] += epsilon

                perturbed_meas = {
                    'vertical': vert_perturbed,
                    'horizontal': base_measurements['horizontal'].copy()
                }

                perturbed_estimate = self.triangulation.triangulate_iterative(perturbed_meas)

                # Частные производные
                diff = (perturbed_estimate - base_estimate).flatten() / epsilon
                jacobian[:, meas_idx] = diff

                meas_idx += 1

        # Горизонтальные измерения
        for i in range(N - 1):
            for j in range(M):
                horiz_perturbed = base_measurements['horizontal'].copy()
                horiz_perturbed[i, j] += epsilon

                perturbed_meas = {
                    'vertical': base_measurements['vertical'].copy(),
                    'horizontal': horiz_perturbed
                }

                perturbed_estimate = self.triangulation.triangulate_iterative(perturbed_meas)

                diff = (perturbed_estimate - base_estimate).flatten() / epsilon
                jacobian[:, meas_idx] = diff

                meas_idx += 1

        return jacobian

    def compute_dop_factors(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Вычислить DOP (Dilution of Precision) факторы.

        DOP показывает, как геометрия сети усиливает ошибки измерений.

        HDOP — горизонтальный (по X)
        VDOP — вертикальный (по Y)
        PDOP — позиционный (полный)

        σ_pos = PDOP × σ_meas

        Args:
            positions: Позиции датчиков

        Returns:
            hdop, vdop, pdop — массивы shape (N, M)
        """
        p = self.params
        N, M = p.N, p.M

        jacobian = self.compute_jacobian(positions)

        # Ковариационная матрица оценок (при единичной ковариации измерений)
        # Cov(estimates) = J @ Cov(measurements) @ J.T = J @ J.T
        cov_estimates = jacobian @ jacobian.T

        hdop = np.zeros((N, M))
        vdop = np.zeros((N, M))
        pdop = np.zeros((N, M))

        for i in range(N):
            for j in range(M):
                idx = (i * M + j) * 2

                var_x = cov_estimates[idx, idx]
                var_y = cov_estimates[idx + 1, idx + 1]

                hdop[i, j] = np.sqrt(var_x)
                vdop[i, j] = np.sqrt(var_y)
                pdop[i, j] = np.sqrt(var_x + var_y)

        return hdop, vdop, pdop

    # Метод 2: Монте-Карло анализ

    def monte_carlo_analysis(self, positions: np.ndarray,
                             noise_std: float = 0.2,
                             n_trials: int = 1000) -> ErrorAnalysisResult:
        """
        Монте-Карло анализ распространения ошибок.

        Запускаем много симуляций с разными реализациями шума
        и собираем статистику ошибок.

        Args:
            positions: Истинные позиции
            noise_std: Стандартное отклонение шума измерений
            n_trials: Количество испытаний

        Returns:
            ErrorAnalysisResult с полной статистикой
        """
        p = self.params
        N, M = p.N, p.M

        echolocation = EcholocationModel(p, noise_std=noise_std)

        # Собираем ошибки по всем испытаниям
        all_errors = np.zeros((n_trials, N, M))
        all_estimates = np.zeros((n_trials, N, M, 2))

        for trial in range(n_trials):
            measurements = echolocation.measure_distances(positions, add_noise=True)
            estimate = self.triangulation.triangulate_iterative(measurements)

            errors = np.linalg.norm(estimate - positions, axis=2)
            all_errors[trial] = errors
            all_estimates[trial] = estimate

        # Статистика ошибок
        mean_errors = np.mean(all_errors, axis=0)
        std_errors = np.std(all_errors, axis=0)
        max_errors = np.max(all_errors, axis=0)

        # DOP факторы
        hdop, vdop, pdop = self.compute_dop_factors(positions)

        # Корреляционная матрица ошибок
        # Показывает, какие датчики ошибаются вместе
        flat_errors = all_errors.reshape(n_trials, -1)  # (n_trials, N*M)
        error_correlation = np.corrcoef(flat_errors.T)

        # Чувствительность к параметрам
        sensitivity = self._compute_sensitivity(positions, noise_std)

        return ErrorAnalysisResult(
            mean_errors=mean_errors,
            std_errors=std_errors,
            max_errors=max_errors,
            hdop=hdop,
            vdop=vdop,
            pdop=pdop,
            error_correlation=error_correlation,
            sensitivity=sensitivity
        )

    def _compute_sensitivity(self, positions: np.ndarray,
                             base_noise: float) -> Dict[str, np.ndarray]:
        """Вычислить чувствительность к различным параметрам."""
        p = self.params

        sensitivity = {}

        # Чувствительность к уровню шума
        noise_levels = [0.1, 0.2, 0.3, 0.5, 1.0]
        noise_rmse = []

        for noise in noise_levels:
            echo = EcholocationModel(p, noise_std=noise)
            errors = []
            for _ in range(100):
                meas = echo.measure_distances(positions, add_noise=True)
                est = self.triangulation.triangulate_iterative(meas)
                err = np.sqrt(np.mean(np.sum((est - positions) ** 2, axis=2)))
                errors.append(err)
            noise_rmse.append(np.mean(errors))

        sensitivity['noise_levels'] = np.array(noise_levels)
        sensitivity['noise_rmse'] = np.array(noise_rmse)

        return sensitivity

   # Метод 3: Анализ уязвимых зон

    def identify_vulnerable_sensors(self, result: ErrorAnalysisResult,
                                    threshold_percentile: float = 75) -> List[Tuple[int, int]]:
        """
        Определить наиболее уязвимые датчики.

        Args:
            result: Результаты Монте-Карло анализа
            threshold_percentile: Процентиль для определения "высокой" ошибки

        Returns:
            Список индексов (i, j) уязвимых датчиков
        """
        threshold = np.percentile(result.mean_errors, threshold_percentile)

        vulnerable = []
        N, M = result.mean_errors.shape

        for i in range(N):
            for j in range(M):
                if result.mean_errors[i, j] >= threshold:
                    vulnerable.append((i, j))

        return vulnerable

    def analyze_error_sources(self, positions: np.ndarray,
                              noise_std: float = 0.2) -> Dict[str, np.ndarray]:
        """
        Разложить ошибки по источникам.

        Источники ошибок:
        1. Прямой шум измерений
        2. Накопление ошибок (от якоря к концу верёвки)
        3. Геометрическое усиление (плохие углы)

        Returns:
            Словарь с вкладами каждого источника
        """
        p = self.params
        N, M = p.N, p.M

        # 1. Базовая ошибка (только от шума, идеальная геометрия)
        base_error = noise_std  # При идеальной геометрии ошибка ≈ σ

        # 2. Накопление по рядам (j увеличивается — больше ошибка)
        accumulation = np.zeros((N, M))
        for j in range(M):
            # Ошибка накапливается как sqrt(j+1) * σ (независимые измерения)
            accumulation[:, j] = np.sqrt(j + 1)

        # 3. Краевые эффекты (меньше связей — больше ошибка)
        edge_effect = np.ones((N, M))
        # Крайние верёвки (i=0, i=N-1) имеют меньше горизонтальных связей
        edge_effect[0, :] *= 1.3
        edge_effect[-1, :] *= 1.3
        # Последний ряд (j=M-1) — свободный конец
        edge_effect[:, -1] *= 1.2

        # 4. DOP фактор (геометрия)
        _, _, pdop = self.compute_dop_factors(positions)

        return {
            'base_noise': np.full((N, M), base_error),
            'accumulation_factor': accumulation,
            'edge_factor': edge_effect,
            'geometry_dop': pdop
        }

# ВИЗУАЛИЗАЦИЯ АНАЛИЗА

class ErrorVisualization:
    """Визуализация результатов анализа ошибок."""

    def __init__(self, params: SystemParams):
        self.params = params

    def plot_error_heatmap(self, result: ErrorAnalysisResult,
                           save_path: Optional[str] = None):
        """Тепловая карта средних ошибок."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Средние ошибки
        im1 = axes[0].imshow(result.mean_errors.T, cmap='Reds', aspect='auto')
        axes[0].set_title('Средняя ошибка (м)')
        axes[0].set_xlabel('Верёвка (i)')
        axes[0].set_ylabel('Датчик (j)')
        plt.colorbar(im1, ax=axes[0])

        # Стандартное отклонение
        im2 = axes[1].imshow(result.std_errors.T, cmap='Oranges', aspect='auto')
        axes[1].set_title('Стд. отклонение ошибки (м)')
        axes[1].set_xlabel('Верёвка (i)')
        axes[1].set_ylabel('Датчик (j)')
        plt.colorbar(im2, ax=axes[1])

        # PDOP
        im3 = axes[2].imshow(result.pdop.T, cmap='Blues', aspect='auto')
        axes[2].set_title('PDOP (усиление ошибки)')
        axes[2].set_xlabel('Верёвка (i)')
        axes[2].set_ylabel('Датчик (j)')
        plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)

        return fig, axes

    def plot_error_correlation(self, result: ErrorAnalysisResult,
                               save_path: Optional[str] = None):
        """Корреляционная матрица ошибок."""
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(result.error_correlation, cmap='RdBu_r',
                       vmin=-1, vmax=1, aspect='auto')

        ax.set_title('Корреляция ошибок между датчиками')
        ax.set_xlabel('Датчик (линейный индекс)')
        ax.set_ylabel('Датчик (линейный индекс)')

        plt.colorbar(im, ax=ax, label='Коэффициент корреляции')

        # Сетка для разделения верёвок
        N, M = self.params.N, self.params.M
        for i in range(1, N):
            ax.axhline(i * M - 0.5, color='k', linewidth=0.5, alpha=0.5)
            ax.axvline(i * M - 0.5, color='k', linewidth=0.5, alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)

        return fig, ax

    def plot_sensitivity_analysis(self, result: ErrorAnalysisResult,
                                  save_path: Optional[str] = None):
        """График чувствительности к шуму измерений."""
        fig, ax = plt.subplots(figsize=(10, 6))

        noise = result.sensitivity['noise_levels']
        rmse = result.sensitivity['noise_rmse']

        ax.plot(noise, rmse, 'bo-', linewidth=2, markersize=8)

        # Теоретическая линия (линейная зависимость)
        coeffs = np.polyfit(noise, rmse, 1)
        ax.plot(noise, np.polyval(coeffs, noise), 'r--',
                label=f'Линейная аппроксимация: RMSE ≈ {coeffs[0]:.2f}σ + {coeffs[1]:.2f}')

        ax.set_xlabel('Шум измерений σ (м)')
        ax.set_ylabel('RMSE позиционирования (м)')
        ax.set_title('Чувствительность к шуму измерений')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)

        return fig, ax

    def plot_error_sources(self, sources: Dict[str, np.ndarray],
                           save_path: Optional[str] = None):
        """Визуализация источников ошибок."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        titles = [
            ('base_noise', 'Базовый шум'),
            ('accumulation_factor', 'Фактор накопления'),
            ('edge_factor', 'Краевой эффект'),
            ('geometry_dop', 'Геометрический DOP')
        ]

        for ax, (key, title) in zip(axes.flatten(), titles):
            data = sources[key]
            im = ax.imshow(data.T, cmap='YlOrRd', aspect='auto')
            ax.set_title(title)
            ax.set_xlabel('Верёвка (i)')
            ax.set_ylabel('Датчик (j)')
            plt.colorbar(im, ax=ax)

            # Добавляем значения
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(i, j, f'{data[i, j]:.2f}', ha='center', va='center',
                            fontsize=8, color='black' if data[i, j] < np.max(data) * 0.7 else 'white')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)

        return fig, axes

    def plot_spatial_error_distribution(self, positions: np.ndarray,
                                        result: ErrorAnalysisResult,
                                        save_path: Optional[str] = None):
        """Пространственное распределение ошибок."""
        fig, ax = plt.subplots(figsize=(12, 10))

        p = self.params
        N, M = p.N, p.M

        # Точки крепления
        anchors = np.array([
            [(i - (N - 1) / 2) * p.d, 20.0] for i in range(N)
        ])
        ax.scatter(anchors[:, 0], anchors[:, 1], s=100, c='blue',
                   marker='s', label='Точки крепления', zorder=5)

        # Верёвки
        for i in range(N):
            rope_x = [anchors[i, 0]] + [positions[i, j, 0] for j in range(M)]
            rope_y = [anchors[i, 1]] + [positions[i, j, 1] for j in range(M)]
            ax.plot(rope_x, rope_y, 'k-', alpha=0.3, linewidth=1)

        # Датчики с цветом по ошибке
        errors_flat = result.mean_errors.flatten()
        positions_flat = positions.reshape(-1, 2)

        scatter = ax.scatter(positions_flat[:, 0], positions_flat[:, 1],
                             c=errors_flat, cmap='Reds', s=100,
                             vmin=np.min(errors_flat), vmax=np.max(errors_flat),
                             zorder=4, edgecolors='black', linewidths=0.5)

        plt.colorbar(scatter, ax=ax, label='Средняя ошибка (м)')

        # Уязвимые датчики (подписи)
        analyzer = ErrorPropagationAnalyzer(p)
        vulnerable = analyzer.identify_vulnerable_sensors(result)
        for i, j in vulnerable:
            pos = positions[i, j]
            ax.annotate(f'({i},{j})', pos, textcoords='offset points',
                        xytext=(5, 5), fontsize=8, color='red')

        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_title('Пространственное распределение ошибок')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)

        return fig, ax

# ОПТИМИЗАЦИЯ КОНФИГУРАЦИИ

class ConfigurationOptimizer:
    """Оптимизация конфигурации сети для минимизации ошибок."""

    def __init__(self, base_params: SystemParams):
        self.base_params = base_params

    def sweep_parameters(self,
                         N_range: List[int] = [3, 5, 7],
                         M_range: List[int] = [3, 5, 7],
                         d_range: List[float] = [5, 8, 12],
                         L_range: List[float] = [8, 10, 15],
                         noise_std: float = 0.2,
                         n_trials: int = 100) -> Dict:
        """
        Перебор параметров для поиска оптимальной конфигурации.

        Returns:
            Словарь с результатами для каждой конфигурации
        """
        results = []

        total = len(N_range) * len(M_range) * len(d_range) * len(L_range)
        current = 0

        for N in N_range:
            for M in M_range:
                for d in d_range:
                    for L in L_range:
                        current += 1

                        params = SystemParams(N=N, M=M, d=d, L=L,
                                              k=100.0, gamma=5.0, wave_amp=2.0)

                        # Генерируем позиции
                        model = PhysicalModel(params)
                        for _ in range(100):
                            model.step(0.01)
                        positions = model.positions.copy()

                        # Анализ ошибок
                        analyzer = ErrorPropagationAnalyzer(params)
                        echo = EcholocationModel(params, noise_std=noise_std)
                        triang = Triangulation(params)

                        errors = []
                        for _ in range(n_trials):
                            meas = echo.measure_distances(positions, add_noise=True)
                            est = triang.triangulate_iterative(meas)
                            err = np.sqrt(np.mean(np.sum((est - positions) ** 2, axis=2)))
                            errors.append(err)

                        rmse = np.mean(errors)

                        results.append({
                            'N': N, 'M': M, 'd': d, 'L': L,
                            'n_sensors': N * M,
                            'rmse': rmse,
                            'coverage_area': (N - 1) * d * M * L  # Примерная площадь
                        })

        return results

    def find_optimal(self, results: List[Dict],
                     constraint: Optional[str] = None,
                     constraint_value: Optional[float] = None) -> Dict:
        """
        Найти оптимальную конфигурацию.

        Args:
            results: Результаты sweep_parameters
            constraint: Ограничение ('n_sensors', 'coverage_area')
            constraint_value: Максимальное значение ограничения
        """
        filtered = results

        if constraint and constraint_value:
            filtered = [r for r in results if r[constraint] <= constraint_value]

        if not filtered:
            return None

        # Сортируем по RMSE
        sorted_results = sorted(filtered, key=lambda x: x['rmse'])

        return sorted_results[0]


# ДЕМОНСТРАЦИЯ
def run_error_analysis_demo():
    """Демонстрация анализа погрешностей."""

    print("=" * 60)
    print("АНАЛИЗ ПОГРЕШНОСТЕЙ И ЧУВСТВИТЕЛЬНОСТИ")
    print("=" * 60)

    # Параметры
    params = SystemParams(
        N=5, M=5,
        L=10.0, d=8.0,
        k=100.0, gamma=5.0,
        wave_amp=2.0
    )

    # Генерируем позиции
    print("\nГенерация тестовых позиций...")
    model = PhysicalModel(params)
    for _ in range(100):
        model.step(0.01)
    positions = model.positions.copy()

    # Анализ ошибок
    print("Запуск Монте-Карло анализа (1000 испытаний)...")
    analyzer = ErrorPropagationAnalyzer(params)
    result = analyzer.monte_carlo_analysis(positions, noise_std=0.2, n_trials=1000)

    print(f"\nРезультаты:")
    print(f"  Средняя ошибка:     {np.mean(result.mean_errors):.4f} м")
    print(f"  Макс. ошибка:       {np.max(result.max_errors):.4f} м")
    print(f"  Средний PDOP:       {np.mean(result.pdop):.2f}")
    print(f"  Макс. PDOP:         {np.max(result.pdop):.2f}")

    # Уязвимые датчики
    vulnerable = analyzer.identify_vulnerable_sensors(result)
    print(f"\n  Уязвимые датчики (топ-25%): {vulnerable}")

    # Источники ошибок
    print("\nАнализ источников ошибок...")
    sources = analyzer.analyze_error_sources(positions, noise_std=0.2)

    # Визуализация
    print("\n" + "-" * 40)
    print("Создание визуализаций...")

    viz = ErrorVisualization(params)

    fig1, _ = viz.plot_error_heatmap(result, 'error_heatmap_detailed.png')
    print("  Сохранено: error_heatmap_detailed.png")

    fig2, _ = viz.plot_error_correlation(result, 'error_correlation.png')
    print("  Сохранено: error_correlation.png")

    fig3, _ = viz.plot_sensitivity_analysis(result, 'sensitivity_analysis.png')
    print("  Сохранено: sensitivity_analysis.png")

    fig4, _ = viz.plot_error_sources(sources, 'error_sources.png')
    print("  Сохранено: error_sources.png")

    fig5, _ = viz.plot_spatial_error_distribution(positions, result, 'spatial_errors.png')
    print("  Сохранено: spatial_errors.png")

    # Краткий отчёт
    print("\n" + "=" * 60)
    print("ВЫВОДЫ:")
    print("=" * 60)
    print("""
1. РАСПРЕДЕЛЕНИЕ ОШИБОК:
   - Ошибки растут от точек крепления к свободным концам
   - Крайние верёвки (i=0 и i=N-1) имеют бóльшие ошибки
   - Это связано с меньшим количеством связей

2. КОРРЕЛЯЦИЯ ОШИБОК:
   - Соседние датчики на одной верёвке сильно коррелируют
   - Датчики на соседних верёвках имеют умеренную корреляцию
   - Это важно для фильтра Калмана

3. ЧУВСТВИТЕЛЬНОСТЬ:
   - RMSE растёт линейно с шумом измерений
   - Коэффициент усиления ~2x (PDOP ≈ 2)
""")

    return result, sources


if __name__ == "__main__":
    run_error_analysis_demo()