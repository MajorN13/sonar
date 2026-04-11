import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle, Polygon
from typing import Optional, Dict, List

from Simulation1 import SystemParams, PhysicalModel, EcholocationModel, Triangulation


# БАЗОВАЯ АНИМАЦИЯ
class SensorAnimator:
    """
    Класс для создания анимаций системы датчиков.
    """

    def __init__(self, params: Optional[SystemParams] = None,
                 ship_gps: tuple = (26.629167, -70.883611)):
        """
        Args:
            params: Параметры системы (если None, используются значения по умолчанию)
            ship_gps: GPS координаты корабля (широта, долгота)
        """
        if params is None:
            params = SystemParams(
                N=5,  # Количество верёвок (штук)
                M=5,  # Количество датчиков на каждой верёвке (штук)
                L=10.0,  # Длина сегмента верёвки между датчиками (метров)
                d=8.0,  # Расстояние между точками крепления соседних верёвок (метров)
                k=100.0,  # Жёсткость верёвки - коэффициент упругости (Н/м)
                gamma=5.0,  # Коэффициент сопротивления воды (кг/с)
                wave_amp=2.5  # Амплитуда волн - максимальное отклонение (метров)
            )

        self.params = params
        self.ship_gps = ship_gps  # GPS координаты корабля (широта, долгота)
        self.model = PhysicalModel(params)
        self.echolocation = EcholocationModel(params, noise_std=0.0)

    def get_anchors(self) -> np.ndarray:
        """Получить позиции точек крепления в локальной системе координат."""
        p = self.params
        return np.array([
            [(i - (p.N - 1) / 2) * p.d, 20.0]
            for i in range(p.N)
        ])

    def generate_frames(self, duration: float, fps: int,
                        warmup_steps: int = 150) -> List[Dict]:
        """
        Сгенерировать данные для всех кадров.

        Args:
            duration: Продолжительность анимации (секунды)
            fps: Кадров в секунду
            warmup_steps: Количество шагов прогрева модели для стабилизации

        Returns:
            Список словарей с данными для каждого кадра
        """
        # Прогрев модели - даём системе стабилизироваться
        self.model.reset()
        for _ in range(warmup_steps):
            self.model.step(0.01)

        n_frames = int(duration * fps)
        frames = []

        for _ in range(n_frames):
            # Несколько шагов симуляции на кадр для плавности
            for _ in range(2):
                self.model.step(0.01)

            measurements = self.echolocation.measure_distances(
                self.model.positions, add_noise=False
            )

            frames.append({
                'positions': self.model.positions.copy(),
                'time': self.model.time,
                'vertical': measurements['vertical'].copy(),
                'horizontal': measurements['horizontal'].copy()
            })

        return frames

    def create_basic_animation(self,
                               duration: float = 10.0,
                               fps: int = 25,
                               output_file: str = 'sonar_basic.mp4',
                               dpi: int = 120) -> None:
        """
        Создать базовую анимацию движения системы.

        Args:
            duration: Продолжительность анимации (секунды)
            fps: Кадров в секунду
            output_file: Имя выходного файла (.mp4)
            dpi: Разрешение (точек на дюйм)
        """
        print("Генерация кадров...")
        frames = self.generate_frames(duration, fps)

        p = self.params
        anchors = self.get_anchors()

        # Создаём фигуру с белым фоном
        fig, ax = plt.subplots(figsize=(16, 14), facecolor='white')
        ax.set_facecolor('white')

        # Вычисляем границы на основе параметров системы
        x_margin = 25
        y_margin = 20
        x_range = (p.N - 1) * p.d / 2 + x_margin + p.wave_amp
        y_max = 20 + p.M * p.L + y_margin + p.wave_amp

        ax.set_xlim(-x_range, x_range)
        ax.set_ylim(-15, y_max)
        ax.set_aspect('equal')

        # Настройка осей
        ax.set_xlabel('X (м)', fontsize=11)
        ax.set_ylabel('Y (м)', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.tick_params(labelsize=9)

        # Заголовок
        ax.set_title('Буксируемая система эхолокационных датчиков',
                     fontsize=14, fontweight='bold', pad=10)

        # --- Статические элементы ---

        # Корабль
        ship = Rectangle((-12, -2), 24, 12,
                         facecolor='#4a4a4a', edgecolor='black', linewidth=1.5)
        ax.add_patch(ship)
        ax.text(0, 4, 'КОРАБЛЬ', ha='center', va='center',
                color='white', fontsize=10, fontweight='bold')

        # GPS координаты корабля
        gps_text = f'GPS: {self.ship_gps[0]:.4f}°N, {self.ship_gps[1]:.4f}°E'
        ax.text(0, -0.5, gps_text, ha='center', va='center',
                color='white', fontsize=7)

        # Буксировочный трос
        ax.plot([0, 0], [10, 15], color='#333333', linewidth=3)

        # Треугольник (жёсткая рама) - размер зависит от количества верёвок
        tri_half_width = (p.N - 1) * p.d / 2 + 4
        triangle = Polygon([[0, 15], [-tri_half_width, 20], [tri_half_width, 20]],
                           fill=True, facecolor='#808080',
                           edgecolor='black', linewidth=1.5)
        ax.add_patch(triangle)
        ax.text(0, 17, 'Жёсткая рама', ha='center', va='center',
                fontsize=8, color='white')

        # Точки крепления
        ax.scatter(anchors[:, 0], anchors[:, 1], s=120, c='#0066cc',
                   marker='s', zorder=12, edgecolors='black', linewidths=1)

        # Подписи точек крепления
        for i, anchor in enumerate(anchors):
            ax.text(anchor[0], anchor[1] - 2.5, f'K{i + 1}',
                    ha='center', va='top', fontsize=8, fontweight='bold')

        # --- Динамические элементы ---

        # Верёвки
        rope_lines = [ax.plot([], [], color='#666666', linewidth=2,
                              zorder=6)[0] for _ in range(p.N)]

        # Датчики
        sensor_scatter = ax.scatter([], [], s=80, c='#00aa00', marker='o',
                                    zorder=11, edgecolors='black', linewidths=1)

        # Горизонтальные связи эхолокации
        echo_h_lines = [ax.plot([], [], color='#ff6600', linewidth=1.5,
                                linestyle='--', zorder=5)[0]
                        for _ in range((p.N - 1) * p.M)]

        # Вертикальные связи эхолокации
        echo_v_lines = [ax.plot([], [], color='#0088ff', linewidth=1,
                                linestyle=':', zorder=4)[0]
                        for _ in range(p.N * p.M)]

        # Тексты расстояний - горизонтальные (между верёвками)
        dist_h_texts = []
        for i in range(p.N - 1):
            for j in range(p.M):
                t = ax.text(0, 0, '', fontsize=7, color='#cc5500',
                            ha='center', va='bottom', fontweight='bold')
                dist_h_texts.append(t)

        # Тексты расстояний - вертикальные (вдоль верёвок)
        dist_v_texts = []
        for i in range(p.N):
            for j in range(p.M):
                t = ax.text(0, 0, '', fontsize=6, color='#0066cc',
                            ha='left', va='center')
                dist_v_texts.append(t)

        # Время
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                            fontsize=11, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white',
                                      edgecolor='gray', alpha=0.9))

        # Информационная панель с параметрами
        info_text = (
            f'ПАРАМЕТРЫ СИСТЕМЫ:\n'
            f'Верёвок N = {p.N} шт\n'
            f'Датчиков на верёвке M = {p.M} шт\n'
            f'Всего датчиков: {p.N * p.M} шт\n'
            f'Шаг верёвки L = {p.L} м\n'
            f'Шаг крепления d = {p.d} м\n'
            f'Жёсткость k = {p.k} Н/м\n'
            f'Сопротивление воды γ = {p.gamma} кг/с\n'
            f'Амплитуда волн A = {p.wave_amp} м'
        )
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white',
                          edgecolor='gray', alpha=0.9),
                family='monospace')

        # Легенда
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#0066cc',
                   markersize=10, label='Точки крепления'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#00aa00',
                   markersize=8, label='Датчики'),
            Line2D([0], [0], color='#666666', linewidth=2, label='Верёвки'),
            Line2D([0], [0], color='#ff6600', linewidth=1.5, linestyle='--',
                   label='Эхолокация (горизонт.)'),
            Line2D([0], [0], color='#0088ff', linewidth=1, linestyle=':',
                   label='Эхолокация (вертик.)'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

        # Стрелка направления движения
        ax.annotate('', xy=(40, 90), xytext=(40, 75),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
        ax.text(40, 73, 'Направление\nдвижения', ha='center', va='top', fontsize=8)

        def update(frame_idx):
            data = frames[frame_idx]
            positions = data['positions']
            t = data['time']
            vert_dists = data['vertical']
            horiz_dists = data['horizontal']

            # Датчики
            all_pos = positions.reshape(-1, 2)
            sensor_scatter.set_offsets(all_pos)

            # Верёвки
            for i in range(p.N):
                rx = [anchors[i, 0]] + list(positions[i, :, 0])
                ry = [anchors[i, 1]] + list(positions[i, :, 1])
                rope_lines[i].set_data(rx, ry)

            # Горизонтальная эхолокация и расстояния
            idx = 0
            for i in range(p.N - 1):
                for j in range(p.M):
                    # Линия связи
                    echo_h_lines[idx].set_data(
                        [positions[i, j, 0], positions[i + 1, j, 0]],
                        [positions[i, j, 1], positions[i + 1, j, 1]]
                    )
                    # Текст расстояния
                    mx = (positions[i, j, 0] + positions[i + 1, j, 0]) / 2
                    my = (positions[i, j, 1] + positions[i + 1, j, 1]) / 2
                    dist_h_texts[idx].set_position((mx, my + 1))
                    dist_h_texts[idx].set_text(f'{horiz_dists[i, j]:.1f}')
                    idx += 1

            # Вертикальная эхолокация и расстояния
            idx = 0
            for i in range(p.N):
                for j in range(p.M):
                    if j == 0:
                        # От крепления до первого датчика
                        echo_v_lines[idx].set_data(
                            [anchors[i, 0], positions[i, 0, 0]],
                            [anchors[i, 1], positions[i, 0, 1]]
                        )
                        mx = (anchors[i, 0] + positions[i, 0, 0]) / 2
                        my = (anchors[i, 1] + positions[i, 0, 1]) / 2
                    else:
                        # Между соседними датчиками
                        echo_v_lines[idx].set_data(
                            [positions[i, j - 1, 0], positions[i, j, 0]],
                            [positions[i, j - 1, 1], positions[i, j, 1]]
                        )
                        mx = (positions[i, j - 1, 0] + positions[i, j, 0]) / 2
                        my = (positions[i, j - 1, 1] + positions[i, j, 1]) / 2

                    # Подписи только для первой верёвки (чтобы не загромождать)
                    if i == 0:
                        dist_v_texts[idx].set_position((mx - 3, my))
                        dist_v_texts[idx].set_text(f'{vert_dists[i, j]:.1f}м')
                    else:
                        dist_v_texts[idx].set_text('')
                    idx += 1

            time_text.set_text(f't = {t:.1f} с')

            return ([sensor_scatter, time_text] + rope_lines +
                    echo_h_lines + echo_v_lines + dist_h_texts + dist_v_texts)

        print("Создание анимации...")
        anim = FuncAnimation(fig, update, frames=len(frames),
                             blit=True, interval=1000 / fps)

        print(f"Сохранение в {output_file}...")
        writer = FFMpegWriter(fps=fps, bitrate=3000)
        anim.save(output_file, writer=writer, dpi=dpi)

        print(f"Готово! Анимация сохранена в {output_file}")
        plt.close()


# АНИМАЦИЯ С ТРИАНГУЛЯЦИЕЙ

class TriangulationAnimator(SensorAnimator):
    """
    Показывает истинные позиции датчиков и восстановленные
    по измерениям эхолокации.
    """

    def __init__(self, params: Optional[SystemParams] = None,
                 ship_gps: tuple = (26.629167, -70.883611),
                 noise_std: float = 0.2):
        """
        Args:
            params: Параметры системы
            ship_gps: GPS координаты корабля (широта, долгота)
            noise_std: Стандартное отклонение шума измерений (метры)
        """
        super().__init__(params, ship_gps)
        self.noise_std = noise_std  # Шум эхолокационных измерений
        self.echolocation_noisy = EcholocationModel(self.params, noise_std=noise_std)
        self.triangulation = Triangulation(self.params)

    def create_triangulation_animation(self,
                                       duration: float = 10.0,
                                       fps: int = 25,
                                       output_file: str = 'sonar_triangulation.mp4',
                                       dpi: int = 120) -> None:
        """
        Создать анимацию с визуализацией процесса триангуляции.

        Показывает:
        - Истинные позиции датчиков (зелёные круги)
        - Восстановленные позиции (красные крестики)
        - Линии ошибок между ними
        - RMSE и максимальную ошибку в реальном времени

        Args:
            duration: Продолжительность анимации (секунды)
            fps: Кадров в секунду
            output_file: Имя выходного файла (.mp4)
            dpi: Разрешение
        """
        print("Генерация кадров...")

        # Прогрев модели
        self.model.reset()
        for _ in range(150):
            self.model.step(0.01)

        n_frames = int(duration * fps)
        frames = []

        for _ in range(n_frames):
            for _ in range(2):
                self.model.step(0.01)

            true_pos = self.model.positions.copy()

            # Зашумлённые измерения
            measurements = self.echolocation_noisy.measure_distances(true_pos, add_noise=True)

            # Триангуляция - восстановление позиций по измерениям
            estimated_pos = self.triangulation.triangulate_iterative(measurements)

            # Вычисляем ошибки
            errors = np.linalg.norm(estimated_pos - true_pos, axis=2)

            frames.append({
                'true_pos': true_pos,
                'estimated_pos': estimated_pos,
                'errors': errors,
                'time': self.model.time,
                'rmse': np.sqrt(np.mean(errors ** 2)),
                'max_error': np.max(errors)
            })

        p = self.params
        anchors = self.get_anchors()

        # Создаём фигуру с белым фоном
        fig, ax = plt.subplots(figsize=(16, 14), facecolor='white')
        ax.set_facecolor('white')

        # Вычисляем границы на основе параметров системы
        x_margin = 25
        y_margin = 20
        x_range = (p.N - 1) * p.d / 2 + x_margin + p.wave_amp
        y_max = 20 + p.M * p.L + y_margin + p.wave_amp

        ax.set_xlim(-x_range, x_range)
        ax.set_ylim(-15, y_max)
        ax.set_aspect('equal')

        # Настройка осей
        ax.set_xlabel('X (м)', fontsize=11)
        ax.set_ylabel('Y (м)', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.tick_params(labelsize=9)

        # Заголовок
        ax.set_title('Триангуляция эхолокационных датчиков',
                     fontsize=14, fontweight='bold', pad=10)

        # Корабль
        ship = Rectangle((-12, -2), 24, 12,
                         facecolor='#4a4a4a', edgecolor='black', linewidth=1.5)
        ax.add_patch(ship)
        ax.text(0, 4, 'КОРАБЛЬ', ha='center', va='center',
                color='white', fontsize=10, fontweight='bold')
        gps_text = f'GPS: {self.ship_gps[0]:.4f}°N, {self.ship_gps[1]:.4f}°E'
        ax.text(0, -0.5, gps_text, ha='center', va='center',
                color='white', fontsize=7)

        ax.plot([0, 0], [10, 15], color='#333333', linewidth=3)

        # Треугольник - размер зависит от количества верёвок
        tri_half_width = (p.N - 1) * p.d / 2 + 4
        triangle = Polygon([[0, 15], [-tri_half_width, 20], [tri_half_width, 20]],
                           fill=True, facecolor='#808080',
                           edgecolor='black', linewidth=1.5)
        ax.add_patch(triangle)

        ax.scatter(anchors[:, 0], anchors[:, 1], s=120, c='#0066cc',
                   marker='s', zorder=12, edgecolors='black', linewidths=1)

        # Верёвки
        rope_lines = [ax.plot([], [], color='#666666', linewidth=2,
                              zorder=6)[0] for _ in range(p.N)]

        # Истинные позиции (зелёные)
        true_scatter = ax.scatter([], [], s=100, c='#00aa00', marker='o',
                                  zorder=11, edgecolors='black', linewidths=1,
                                  label='Истинные позиции')

        # Восстановленные позиции (красные крестики)
        est_scatter = ax.scatter([], [], s=80, c='#cc0000', marker='x',
                                 zorder=12, linewidths=2,
                                 label='Восстановленные')

        # Линии ошибок
        error_lines = [ax.plot([], [], color='#cc0000', linewidth=1,
                               alpha=0.6, zorder=5)[0]
                       for _ in range(p.N * p.M)]

        # Время и метрики
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                            fontsize=11, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white',
                                      edgecolor='gray', alpha=0.9))

        # Информационная панель
        info_text = (
            f'ПАРАМЕТРЫ:\n'
            f'Датчиков: {p.N}×{p.M} = {p.N * p.M} шт\n'
            f'Шум измерений σ = {self.noise_std} м\n'
            f'──────────────\n'
            f'МЕТРИКИ:\n'
        )
        info_box = ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
                           fontsize=9, verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='white',
                                     edgecolor='gray', alpha=0.9),
                           family='monospace')

        rmse_text = ax.text(0.98, 0.78, '', transform=ax.transAxes,
                            fontsize=10, verticalalignment='top', horizontalalignment='right',
                            color='#006600', fontweight='bold')

        max_err_text = ax.text(0.98, 0.73, '', transform=ax.transAxes,
                               fontsize=10, verticalalignment='top', horizontalalignment='right',
                               color='#cc0000', fontweight='bold')

        # Легенда
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#00aa00',
                   markersize=10, label='Истинные позиции'),
            Line2D([0], [0], marker='x', color='#cc0000', markersize=10,
                   markeredgewidth=2, label='Восстановленные'),
            Line2D([0], [0], color='#cc0000', linewidth=1, label='Ошибка'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

        def update(frame_idx):
            data = frames[frame_idx]
            true_pos = data['true_pos']
            est_pos = data['estimated_pos']
            t = data['time']
            rmse = data['rmse']
            max_err = data['max_error']

            # Истинные позиции
            true_flat = true_pos.reshape(-1, 2)
            true_scatter.set_offsets(true_flat)

            # Восстановленные позиции
            est_flat = est_pos.reshape(-1, 2)
            est_scatter.set_offsets(est_flat)

            # Верёвки (по истинным позициям)
            for i in range(p.N):
                rx = [anchors[i, 0]] + list(true_pos[i, :, 0])
                ry = [anchors[i, 1]] + list(true_pos[i, :, 1])
                rope_lines[i].set_data(rx, ry)

            # Линии ошибок
            idx = 0
            for i in range(p.N):
                for j in range(p.M):
                    error_lines[idx].set_data(
                        [true_pos[i, j, 0], est_pos[i, j, 0]],
                        [true_pos[i, j, 1], est_pos[i, j, 1]]
                    )
                    idx += 1

            time_text.set_text(f't = {t:.1f} с')
            rmse_text.set_text(f'RMSE = {rmse:.3f} м')
            max_err_text.set_text(f'Max = {max_err:.3f} м')

            return ([true_scatter, est_scatter, time_text, rmse_text, max_err_text] +
                    rope_lines + error_lines)

        print("Создание анимации...")
        anim = FuncAnimation(fig, update, frames=len(frames),
                             blit=True, interval=1000 / fps)

        print(f"Сохранение в {output_file}...")
        writer = FFMpegWriter(fps=fps, bitrate=3000)
        anim.save(output_file, writer=writer, dpi=dpi)

        print(f"Готово! Анимация сохранена в {output_file}")
        plt.close()


# ДЕМОНСТРАЦИЯ
def run_animation_demo():
    print("=" * 60)
    print("СОЗДАНИЕ АНИМАЦИЙ СИСТЕМЫ ДАТЧИКОВ")
    print("=" * 60)

    # Параметры системы с комментариями
    params = SystemParams(
        N=6,  # Количество верёвок (штук)
        M=7,  # Количество датчиков на каждой верёвке (штук)
        L=10.0,  # Длина сегмента верёвки между датчиками (метров)
        d=8.0,  # Расстояние между точками крепления соседних верёвок (метров)
        k=40.0,  # Жёсткость верёвки - коэффициент упругости (Н/м)
        gamma=3.0,  # Коэффициент сопротивления воды (кг/с)
        wave_amp=12.5  # Амплитуда волн - максимальное отклонение (метров)
    )

    # GPS координаты корабля
    ship_gps = (26.629167, -70.883611)  # широта, долгота

    # 1. Базовая анимация
    print("\n1. Создание базовой анимации...")
    animator = SensorAnimator(params, ship_gps=ship_gps)
    animator.create_basic_animation(
        duration=60.0,
        fps=25,
        output_file='sonar_basic.mp4',
        dpi=120
    )

    # 2. Анимация с триангуляцией
    print("\n2. Создание анимации триангуляции...")
    triang_animator = TriangulationAnimator(
        params,
        ship_gps=ship_gps,
        noise_std=0.3  # Шум измерений эхолокации (метры)
    )
    triang_animator.create_triangulation_animation(
        duration=60.0,
        fps=25,
        output_file='sonar_triangulation.mp4',
        dpi=120
    )

    print("\n" + "=" * 60)
    print("Все анимации созданы!")
    print("=" * 60)


if __name__ == "__main__":
    run_animation_demo()