"""
Task A: Customer Pathfinding & Store Layout
Simulación de clientes moviéndose en una tienda usando algoritmos BFS y A*
Siguiendo principios SOLID, KISS y clean code
"""

import pygame
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple
from collections import deque
import heapq
import random


# ======================= CONSTANTS =======================
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
GRID_ROWS = 8
GRID_COLS = 6
CELL_SIZE = 70
GRID_OFFSET_X = 50
GRID_OFFSET_Y = 50

# Simulation parameters
CUSTOMER_MOVE_SPEED = 10  # Frames entre movimientos (mayor = más lento)
MIN_STAY_TIME = 30  # Mínimo de frames para quedarse en una celda
MAX_STAY_TIME = 120  # Máximo de frames para quedarse en una celda
STAY_PROBABILITY = 0.25  # Probabilidad de quedarse en una celda (25%)

# Colors
COLOR_BG = (240, 240, 240)
COLOR_GRID = (200, 200, 200)
COLOR_WALKABLE = (255, 255, 255)
COLOR_BEAUTY = (255, 182, 193)      # Pink
COLOR_CLOTHING = (173, 216, 230)    # Light Blue
COLOR_ELECTRONICS = (255, 255, 153) # Light Yellow
COLOR_ENTRANCE = (144, 238, 144)    # Light Green
COLOR_CUSTOMER = (255, 69, 0)       # Red-Orange
COLOR_BUTTON = (70, 130, 180)       # Steel Blue
COLOR_BUTTON_HOVER = (100, 149, 237)
COLOR_TEXT = (0, 0, 0)
COLOR_HEATMAP_LOW = (255, 255, 255)
COLOR_HEATMAP_HIGH = (255, 0, 0)


# ======================= ENUMS =======================
class CellType(Enum):
    """Tipos de celdas en la tienda"""
    WALKABLE = "walkable"
    BEAUTY = "beauty"
    CLOTHING = "clothing"
    ELECTRONICS = "electronics"
    ENTRANCE = "entrance"


class SearchAlgorithm(Enum):
    """Algoritmos de búsqueda disponibles"""
    BFS = "BFS"
    A_STAR = "A*"


# ======================= DATA CLASSES =======================
@dataclass
class Position:
    """Representa una posición en el grid"""
    row: int
    col: int

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        return isinstance(other, Position) and self.row == other.row and self.col == other.col

    def to_tuple(self) -> Tuple[int, int]:
        return (self.row, self.col)


# ======================= DATA LOADER =======================
class DataLoader:
    """Responsable de cargar y analizar los datos del CSV"""

    @staticmethod
    def load_sales_data(filepath: str) -> pd.DataFrame:
        """Carga los datos de ventas desde el CSV"""
        try:
            df = pd.read_csv(filepath)
            return df
        except FileNotFoundError:
            print(f"Warning: {filepath} not found. Using default configuration.")
            return pd.DataFrame()

    @staticmethod
    def get_category_frequencies(df: pd.DataFrame) -> dict:
        """Obtiene la frecuencia de compra por categoría"""
        if df.empty:
            return {'Beauty': 1, 'Clothing': 1, 'Electronics': 1}

        category_counts = df['Product Category'].value_counts().to_dict()
        return category_counts


# ======================= STORE GRID =======================
class StoreGrid:
    """Representa el layout de la tienda como un grid"""

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.grid = [[CellType.WALKABLE for _ in range(cols)] for _ in range(rows)]
        self.entrance = Position(rows - 1, cols // 2)
        self.traffic_count = [[0 for _ in range(cols)] for _ in range(rows)]
        self._initialize_store_layout()

    def _initialize_store_layout(self):
        """Inicializa el layout de la tienda con secciones de productos"""
        # Entrance
        self.grid[self.entrance.row][self.entrance.col] = CellType.ENTRANCE

        # Beauty section (top-left)
        for r in range(0, 3):
            for c in range(0, 2):
                self.grid[r][c] = CellType.BEAUTY

        # Clothing section (top-right)
        for r in range(0, 3):
            for c in range(4, 6):
                self.grid[r][c] = CellType.CLOTHING

        # Electronics section (middle)
        for r in range(4, 6):
            for c in range(2, 4):
                self.grid[r][c] = CellType.ELECTRONICS

    def is_valid_position(self, pos: Position) -> bool:
        """Verifica si una posición es válida"""
        return 0 <= pos.row < self.rows and 0 <= pos.col < self.cols

    def get_cell_type(self, pos: Position) -> CellType:
        """Obtiene el tipo de celda en una posición"""
        if not self.is_valid_position(pos):
            return None
        return self.grid[pos.row][pos.col]

    def get_neighbors(self, pos: Position) -> List[Position]:
        """Obtiene las posiciones vecinas válidas (4-direcciones)"""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        neighbors = []

        for dr, dc in directions:
            new_pos = Position(pos.row + dr, pos.col + dc)
            if self.is_valid_position(new_pos):
                neighbors.append(new_pos)

        return neighbors

    def get_product_sections(self) -> dict:
        """Obtiene todas las posiciones de cada sección de producto"""
        sections = {
            CellType.BEAUTY: [],
            CellType.CLOTHING: [],
            CellType.ELECTRONICS: []
        }

        for r in range(self.rows):
            for c in range(self.cols):
                cell_type = self.grid[r][c]
                if cell_type in sections:
                    sections[cell_type].append(Position(r, c))

        return sections

    def increment_traffic(self, pos: Position):
        """Incrementa el contador de tráfico en una posición"""
        if self.is_valid_position(pos):
            self.traffic_count[pos.row][pos.col] += 1

    def get_max_traffic(self) -> int:
        """Obtiene el máximo tráfico registrado"""
        return max(max(row) for row in self.traffic_count)


# ======================= PATHFINDING =======================
class PathFinder:
    """Clase base para algoritmos de pathfinding"""

    def __init__(self, grid: StoreGrid):
        self.grid = grid

    def find_path(self, start: Position, goal: Position) -> List[Position]:
        """Método abstracto para encontrar un camino"""
        raise NotImplementedError


class BFSPathFinder(PathFinder):
    """Implementación de BFS (Breadth-First Search)"""

    def find_path(self, start: Position, goal: Position) -> List[Position]:
        """Encuentra el camino usando BFS"""
        if not self.grid.is_valid_position(start) or not self.grid.is_valid_position(goal):
            return []

        queue = deque([start])
        visited = {start}
        parent = {start: None}

        while queue:
            current = queue.popleft()

            if current == goal:
                return self._reconstruct_path(parent, start, goal)

            for neighbor in self.grid.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)

        return []

    def _reconstruct_path(self, parent: dict, start: Position, goal: Position) -> List[Position]:
        """Reconstruye el camino desde el diccionario de padres"""
        path = []
        current = goal

        while current is not None:
            path.append(current)
            current = parent[current]

        path.reverse()
        return path


class AStarPathFinder(PathFinder):
    """Implementación de A* (A-Star)"""

    def find_path(self, start: Position, goal: Position) -> List[Position]:
        """Encuentra el camino usando A*"""
        if not self.grid.is_valid_position(start) or not self.grid.is_valid_position(goal):
            return []

        open_set = []
        heapq.heappush(open_set, (0, id(start), start))

        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}

        open_set_hash = {start}

        while open_set:
            _, _, current = heapq.heappop(open_set)
            open_set_hash.discard(current)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            for neighbor in self.grid.get_neighbors(current):
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal)

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], id(neighbor), neighbor))
                        open_set_hash.add(neighbor)

        return []

    def _heuristic(self, pos1: Position, pos2: Position) -> float:
        """Calcula la distancia Manhattan entre dos posiciones"""
        return abs(pos1.row - pos2.row) + abs(pos1.col - pos2.col)

    def _reconstruct_path(self, came_from: dict, current: Position) -> List[Position]:
        """Reconstruye el camino"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


# ======================= CUSTOMER =======================
class Customer:
    """Representa un cliente moviéndose por la tienda"""

    def __init__(self, customer_id: int, start_pos: Position, target_section: CellType,
                 pathfinder: PathFinder, grid: StoreGrid):
        self.id = customer_id
        self.position = start_pos
        self.target_section = target_section
        self.pathfinder = pathfinder
        self.grid = grid
        self.path = []
        self.path_index = 0
        self.finished = False
        self.target_pos = None
        self.stay_timer = 0
        self.move_counter = 0  # Contador para controlar velocidad de movimiento
        self._calculate_path()

    def _calculate_path(self):
        """Calcula el camino hacia una sección objetivo"""
        sections = self.grid.get_product_sections()
        target_positions = sections.get(self.target_section, [])

        if target_positions:
            self.target_pos = random.choice(target_positions)
            self.path = self.pathfinder.find_path(self.position, self.target_pos)
            self.path_index = 0

    def update(self):
        """Actualiza la posición del cliente"""
        if self.finished or not self.path:
            return

        # Comportamiento de permanencia (cliente mirando productos)
        if self.stay_timer > 0:
            self.stay_timer -= 1
            return

        # Control de velocidad de movimiento
        self.move_counter += 1
        if self.move_counter < CUSTOMER_MOVE_SPEED:
            return

        # Resetear contador de movimiento
        self.move_counter = 0

        if self.path_index < len(self.path):
            self.position = self.path[self.path_index]
            self.grid.increment_traffic(self.position)
            self.path_index += 1

            # Decidir si el cliente se queda en la celda (mirando productos)
            if random.random() < STAY_PROBABILITY:
                self.stay_timer = random.randint(MIN_STAY_TIME, MAX_STAY_TIME)
        else:
            self.finished = True

    def recalculate_path(self, new_pathfinder: PathFinder):
        """Recalcula el camino con un nuevo pathfinder"""
        self.pathfinder = new_pathfinder
        if not self.finished and self.target_pos:
            self.path = self.pathfinder.find_path(self.position, self.target_pos)
            self.path_index = 0


# ======================= CUSTOMER MANAGER =======================
class CustomerManager:
    """Gestiona todos los clientes en la simulación"""

    def __init__(self, grid: StoreGrid, category_weights: dict):
        self.grid = grid
        self.customers: List[Customer] = []
        self.customer_id_counter = 0
        self.category_weights = category_weights
        self.categories = list(category_weights.keys())
        self.weights = list(category_weights.values())

    def spawn_customer(self, pathfinder: PathFinder):
        """Genera un nuevo cliente"""
        start_pos = self.grid.entrance
        target_category = self._select_random_category()

        customer = Customer(
            self.customer_id_counter,
            start_pos,
            target_category,
            pathfinder,
            self.grid
        )

        self.customers.append(customer)
        self.customer_id_counter += 1

    def _select_random_category(self) -> CellType:
        """Selecciona una categoría aleatoria basada en pesos"""
        category_str = random.choices(self.categories, weights=self.weights, k=1)[0]

        category_map = {
            'Beauty': CellType.BEAUTY,
            'Clothing': CellType.CLOTHING,
            'Electronics': CellType.ELECTRONICS
        }

        return category_map.get(category_str, CellType.BEAUTY)

    def update_all(self):
        """Actualiza todos los clientes"""
        for customer in self.customers:
            customer.update()

        # Remover clientes que han terminado
        self.customers = [c for c in self.customers if not c.finished]

    def recalculate_all_paths(self, pathfinder: PathFinder):
        """Recalcula los caminos de todos los clientes"""
        for customer in self.customers:
            customer.recalculate_path(pathfinder)

    def set_customer_count(self, count: int, pathfinder: PathFinder):
        """Establece el número de clientes en la simulación"""
        current_count = len(self.customers)

        if count > current_count:
            # Agregar clientes
            for _ in range(count - current_count):
                self.spawn_customer(pathfinder)
        elif count < current_count:
            # Remover clientes
            self.customers = self.customers[:count]


# ======================= UI COMPONENTS =======================
class Button:
    """Representa un botón clickeable en la UI"""

    def __init__(self, x: int, y: int, width: int, height: int, text: str):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.is_hovered = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Maneja eventos del mouse"""
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.is_hovered:
                return True
        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font):
        """Dibuja el botón"""
        color = COLOR_BUTTON_HOVER if self.is_hovered else COLOR_BUTTON
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        pygame.draw.rect(surface, COLOR_TEXT, self.rect, 2, border_radius=5)

        text_surface = font.render(self.text, True, COLOR_TEXT)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)


# ======================= RENDERER =======================
class Renderer:
    """Responsable de renderizar toda la visualización"""

    def __init__(self, screen: pygame.Surface, grid: StoreGrid):
        self.screen = screen
        self.grid = grid
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)

    def draw_grid(self):
        """Dibuja el grid de la tienda"""
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                x = GRID_OFFSET_X + c * CELL_SIZE
                y = GRID_OFFSET_Y + r * CELL_SIZE

                cell_type = self.grid.grid[r][c]
                color = self._get_cell_color(cell_type)

                pygame.draw.rect(self.screen, color, (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(self.screen, COLOR_GRID, (x, y, CELL_SIZE, CELL_SIZE), 1)

    def _get_cell_color(self, cell_type: CellType) -> Tuple[int, int, int]:
        """Obtiene el color para un tipo de celda"""
        color_map = {
            CellType.WALKABLE: COLOR_WALKABLE,
            CellType.BEAUTY: COLOR_BEAUTY,
            CellType.CLOTHING: COLOR_CLOTHING,
            CellType.ELECTRONICS: COLOR_ELECTRONICS,
            CellType.ENTRANCE: COLOR_ENTRANCE
        }
        return color_map.get(cell_type, COLOR_WALKABLE)

    def draw_heatmap(self):
        """Dibuja el heatmap de tráfico"""
        max_traffic = self.grid.get_max_traffic()
        if max_traffic == 0:
            return

        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                traffic = self.grid.traffic_count[r][c]
                if traffic > 0:
                    x = GRID_OFFSET_X + c * CELL_SIZE
                    y = GRID_OFFSET_Y + r * CELL_SIZE

                    # Calcular intensidad del color
                    intensity = min(traffic / max_traffic, 1.0)
                    alpha = int(150 * intensity)

                    # Crear superficie semi-transparente
                    overlay = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                    overlay.fill((255, 0, 0, alpha))
                    self.screen.blit(overlay, (x, y))

    def draw_customers(self, customers: List[Customer]):
        """Dibuja todos los clientes"""
        for customer in customers:
            pos = customer.position
            x = GRID_OFFSET_X + pos.col * CELL_SIZE + CELL_SIZE // 2
            y = GRID_OFFSET_Y + pos.row * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(self.screen, COLOR_CUSTOMER, (x, y), 8)

    def draw_labels(self):
        """Dibuja las etiquetas de las secciones"""
        labels = [
            ("Beauty", GRID_OFFSET_X + CELL_SIZE, GRID_OFFSET_Y + CELL_SIZE),
            ("Clothing", GRID_OFFSET_X + 5 * CELL_SIZE, GRID_OFFSET_Y + CELL_SIZE),
            ("Electronics", GRID_OFFSET_X + 3 * CELL_SIZE, GRID_OFFSET_Y + 5 * CELL_SIZE),
            ("ENTRANCE", GRID_OFFSET_X + 3 * CELL_SIZE, GRID_OFFSET_Y + 7 * CELL_SIZE + 20)
        ]

        for text, x, y in labels:
            text_surface = self.font_small.render(text, True, COLOR_TEXT)
            text_rect = text_surface.get_rect(center=(x, y))
            self.screen.blit(text_surface, text_rect)

    def draw_info(self, algorithm: SearchAlgorithm, customer_count: int):
        """Dibuja información de la simulación"""
        info_x = GRID_OFFSET_X + GRID_COLS * CELL_SIZE + 50
        info_y = GRID_OFFSET_Y

        texts = [
            f"Algorithm: {algorithm.value}",
            f"Customers: {customer_count}",
            "",
            "High-Traffic Areas:",
            "(Shown in red overlay)"
        ]

        for i, text in enumerate(texts):
            text_surface = self.font_small.render(text, True, COLOR_TEXT)
            self.screen.blit(text_surface, (info_x, info_y + i * 25))


# ======================= SIMULATION =======================
class Simulation:
    """Clase principal que gestiona la simulación"""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Task A: Customer Pathfinding & Store Layout")
        self.clock = pygame.time.Clock()
        self.running = True

        # Cargar datos
        data_loader = DataLoader()
        df = data_loader.load_sales_data("retail_sales_dataset.csv")
        category_weights = data_loader.get_category_frequencies(df)

        # Inicializar componentes
        self.grid = StoreGrid(GRID_ROWS, GRID_COLS)
        self.renderer = Renderer(self.screen, self.grid)
        self.customer_manager = CustomerManager(self.grid, category_weights)

        # Estado de la simulación
        self.current_algorithm = SearchAlgorithm.BFS
        self.pathfinder = BFSPathFinder(self.grid)
        self.customer_counts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        self.current_customer_index = 0

        # UI
        button_y = GRID_OFFSET_Y + GRID_ROWS * CELL_SIZE + 30
        self.algorithm_button = Button(
            GRID_OFFSET_X, button_y, 200, 40,
            f"Algorithm: {self.current_algorithm.value}"
        )
        self.customer_button = Button(
            GRID_OFFSET_X + 220, button_y, 200, 40,
            f"Customers: {self.customer_counts[self.current_customer_index]}"
        )

        # Inicializar clientes
        self._initialize_customers()

        # Timer para spawning
        self.update_counter = 0

    def _initialize_customers(self):
        """Inicializa los clientes en la simulación"""
        count = self.customer_counts[self.current_customer_index]
        self.customer_manager.set_customer_count(count, self.pathfinder)

    def toggle_algorithm(self):
        """Cambia el algoritmo de búsqueda"""
        if self.current_algorithm == SearchAlgorithm.BFS:
            self.current_algorithm = SearchAlgorithm.A_STAR
            self.pathfinder = AStarPathFinder(self.grid)
        else:
            self.current_algorithm = SearchAlgorithm.BFS
            self.pathfinder = BFSPathFinder(self.grid)

        self.algorithm_button.text = f"Algorithm: {self.current_algorithm.value}"
        self.customer_manager.recalculate_all_paths(self.pathfinder)

    def toggle_customer_count(self):
        """Cambia la cantidad de clientes"""
        self.current_customer_index = (self.current_customer_index + 1) % len(self.customer_counts)
        count = self.customer_counts[self.current_customer_index]
        self.customer_button.text = f"Customers: {count}"
        self.customer_manager.set_customer_count(count, self.pathfinder)

    def handle_events(self):
        """Maneja los eventos de pygame"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if self.algorithm_button.handle_event(event):
                self.toggle_algorithm()

            if self.customer_button.handle_event(event):
                self.toggle_customer_count()

    def update(self):
        """Actualiza el estado de la simulación"""
        self.update_counter += 1

        # Actualizar clientes cada frame
        self.customer_manager.update_all()

        # Spawn nuevos clientes si es necesario
        if self.update_counter % 30 == 0:  # Cada 30 frames
            target_count = self.customer_counts[self.current_customer_index]
            current_count = len(self.customer_manager.customers)

            if current_count < target_count:
                # spwan de varios clientes a la vez pero sin superar el target
                for _ in range(min(3, target_count - current_count)):
                    self.customer_manager.spawn_customer(self.pathfinder)

    def render(self):
        """Renderiza la simulación"""
        self.screen.fill(COLOR_BG)

        # Dibujar grid y elementos
        self.renderer.draw_grid()
        self.renderer.draw_heatmap()
        self.renderer.draw_customers(self.customer_manager.customers)
        self.renderer.draw_labels()

        # Dibujar UI
        self.algorithm_button.draw(self.screen, self.renderer.font)
        self.customer_button.draw(self.screen, self.renderer.font)
        self.renderer.draw_info(
            self.current_algorithm,
            len(self.customer_manager.customers)
        )

        pygame.display.flip()

    def run(self):
        """Loop principal de la simulación"""
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(60)  # 60 FPS

        pygame.quit()


# ======================= MAIN =======================
def main():
    """Punto de entrada principal"""
    simulation = Simulation()
    simulation.run()


if __name__ == "__main__":
    main()
