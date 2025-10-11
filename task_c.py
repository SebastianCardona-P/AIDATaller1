"""
Task C: Dynamic Pricing with Reinforcement Learning
Sistema de ajuste dinámico de precios usando RL (PPO) para maximizar ingresos.

Arquitectura:
- DataLoader: Carga y procesa datos del CSV
- PricingEnvironment: Ambiente Gymnasium personalizado
- RLAgent: Agente PPO de Stable-Baselines3
- Visualizer: Interfaz Pygame con gráficas en tiempo real
"""

import pygame
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
from collections import deque

# ======================= CONFIGURACIÓN =======================
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 60

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
BLUE = (70, 130, 180)
GREEN = (50, 205, 50)
RED = (220, 20, 60)
ORANGE = (255, 140, 0)
LIGHT_BLUE = (173, 216, 230)

# Parámetros RL
HISTORY_LENGTH = 10  # Ventana de observación de ventas
MAX_PRICE_MULTIPLIER = 2.0  # Máximo incremento de precio
MIN_PRICE_MULTIPLIER = 0.5  # Mínimo descuento
TRAINING_TIMESTEPS = 50000  # Pasos de entrenamiento


# ======================= DATA CLASSES =======================
@dataclass
class ProductData:
    """Datos de un producto"""
    category: str
    base_price: float
    avg_quantity: float
    total_sales: int
    customer_segment: str = "general"


@dataclass
class PricingState:
    """Estado del ambiente de pricing"""
    current_price: float
    base_price: float
    recent_sales: List[float]
    recent_quantities: List[int]
    customer_segment: str
    time_step: int


# ======================= DATA LOADER =======================
class DataLoader:
    """Carga y preprocesa datos del CSV"""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.products: Dict[str, ProductData] = {}

    def load_data(self) -> bool:
        """Carga datos del CSV"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✓ Datos cargados: {len(self.df)} transacciones")
            return True
        except Exception as e:
            print(f"✗ Error cargando datos: {e}")
            return False

    def process_products(self) -> Dict[str, ProductData]:
        """Procesa información de productos por categoría"""
        if self.df is None:
            return {}

        categories = self.df['Product Category'].unique()

        for category in categories:
            cat_data = self.df[self.df['Product Category'] == category]

            self.products[category] = ProductData(
                category=category,
                base_price=cat_data['Price per Unit'].mean(),
                avg_quantity=cat_data['Quantity'].mean(),
                total_sales=len(cat_data),
                customer_segment="general"
            )

        print(f"✓ Productos procesados: {list(self.products.keys())}")
        return self.products


# ======================= PRICING ENVIRONMENT =======================
class PricingEnvironment(gym.Env):
    """
    Ambiente Gymnasium para optimización de precios dinámicos.

    State: [precio_actual_normalizado, precio_base, promedio_ventas_recientes,
            std_ventas_recientes, cantidad_promedio, tendencia_ventas]
    Action: ajuste_precio (continuo entre -0.5 y 0.5)
    Reward: revenue - penalizaciones
    """

    def __init__(self, product_data: ProductData, max_steps: int = 100):
        super().__init__()

        self.product_data = product_data
        self.max_steps = max_steps
        self.current_step = 0

        # Historial
        self.sales_history = deque(maxlen=HISTORY_LENGTH)
        self.quantity_history = deque(maxlen=HISTORY_LENGTH)
        self.revenue_history = []

        # Estado inicial
        self.current_price = product_data.base_price
        self.base_price = product_data.base_price

        # Definición de espacios
        # State: [precio_norm, precio_base_norm, avg_ventas, std_ventas, avg_cantidad, tendencia]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
            high=np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Action: ajuste de precio continuo
        self.action_space = spaces.Box(
            low=-0.5, high=0.5, shape=(1,), dtype=np.float32
        )

        # Inicializar historial con ventas base
        for _ in range(HISTORY_LENGTH):
            self.sales_history.append(self._simulate_demand(self.base_price))
            self.quantity_history.append(np.random.poisson(self.product_data.avg_quantity))

    def _get_observation(self) -> np.ndarray:
        """Construye el vector de observación"""
        # Normalizar precio actual
        price_norm = self.current_price / self.base_price
        base_price_norm = 1.0  # El precio base es el punto de referencia

        # Estadísticas de ventas recientes
        if len(self.sales_history) > 0:
            avg_sales = np.mean(self.sales_history)
            std_sales = np.std(self.sales_history) if len(self.sales_history) > 1 else 0.0
            avg_quantity = np.mean(self.quantity_history)

            # Tendencia (diferencia entre promedio reciente y primeras ventas)
            if len(self.sales_history) >= HISTORY_LENGTH:
                recent_avg = np.mean(list(self.sales_history)[-5:])
                old_avg = np.mean(list(self.sales_history)[:5])
                trend = (recent_avg - old_avg) / (old_avg + 1e-6)
                trend = np.clip(trend, -1.0, 1.0)
            else:
                trend = 0.0
        else:
            avg_sales = 0.5
            std_sales = 0.0
            avg_quantity = self.product_data.avg_quantity
            trend = 0.0

        # Normalizar
        avg_sales_norm = np.clip(avg_sales, 0.0, 1.0)
        std_sales_norm = np.clip(std_sales, 0.0, 1.0)
        avg_quantity_norm = np.clip(avg_quantity / (self.product_data.avg_quantity * 2 + 1), 0.0, 1.0)

        return np.array([
            price_norm,
            base_price_norm,
            avg_sales_norm,
            std_sales_norm,
            avg_quantity_norm,
            trend
        ], dtype=np.float32)

    def _simulate_demand(self, price: float) -> float:
        """
        Simula demanda basada en elasticidad de precio.
        Demand = base_demand * (base_price / current_price)^elasticity
        Elasticidad más baja = más ingresos con precios altos
        """
        elasticity = 0.8  # Elasticidad reducida (antes 1.5) - demanda menos sensible al precio
        price_ratio = self.base_price / (price + 1e-6)

        # Demanda base con algo de aleatoriedad
        base_demand = 1.0
        noise = np.random.normal(0, 0.05)  # Menos ruido (antes 0.1)

        # Bonus por precios óptimos (no muy bajos ni muy altos)
        optimal_ratio = 1.15  # Precio óptimo 15% sobre base
        if 1.0 <= (price / self.base_price) <= 1.3:
            demand_bonus = 0.1
        else:
            demand_bonus = 0.0

        demand = base_demand * (price_ratio ** elasticity) + noise + demand_bonus
        return np.clip(demand, 0.1, 3.0)  # Mínimo 0.1 en vez de 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reinicia el ambiente"""
        super().reset(seed=seed)

        self.current_step = 0
        self.current_price = self.base_price
        self.revenue_history = []

        # Reiniciar historial
        self.sales_history.clear()
        self.quantity_history.clear()
        for _ in range(HISTORY_LENGTH):
            self.sales_history.append(self._simulate_demand(self.base_price))
            self.quantity_history.append(np.random.poisson(self.product_data.avg_quantity))

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Ejecuta un paso en el ambiente"""
        # Aplicar acción (ajuste de precio)
        price_adjustment = float(action[0])
        self.current_price = self.base_price * (1.0 + price_adjustment)
        self.current_price = np.clip(
            self.current_price,
            self.base_price * MIN_PRICE_MULTIPLIER,
            self.base_price * MAX_PRICE_MULTIPLIER
        )

        # Simular demanda y ventas
        demand = self._simulate_demand(self.current_price)
        quantity = max(1, int(np.random.poisson(self.product_data.avg_quantity * demand)))
        revenue = self.current_price * quantity

        # Actualizar historial
        self.sales_history.append(demand)
        self.quantity_history.append(quantity)
        self.revenue_history.append(revenue)

        # ===== NUEVA FUNCIÓN DE RECOMPENSA MEJORADA =====
        # Calcular revenue base para comparación justa
        base_demand = self._simulate_demand(self.base_price)
        base_quantity = int(self.product_data.avg_quantity * base_demand)
        base_revenue = self.base_price * base_quantity

        # Reward principal: diferencia de revenue normalizada
        revenue_diff = revenue - base_revenue
        reward = revenue_diff / (base_revenue + 1e-6)

        # Bonus por mantener demanda alta
        if demand > 0.9:
            reward += 0.1

        # Penalización SOLO por precios extremadamente fuera de rango razonable
        price_ratio = self.current_price / self.base_price
        if price_ratio > 1.5 or price_ratio < 0.7:
            # Penalización ligera solo si es muy extremo
            extreme_penalty = (abs(price_ratio - 1.0) - 0.5) * 0.1
            reward -= extreme_penalty

        # Bonus por precios en rango óptimo (1.0 - 1.3x base)
        if 1.0 <= price_ratio <= 1.3:
            reward += 0.15

        # Incrementar paso
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        return self._get_observation(), reward, terminated, truncated, {
            'revenue': revenue,
            'quantity': quantity,
            'price': self.current_price,
            'base_revenue': base_revenue
        }


# ======================= RL AGENT =======================
class RLAgent:
    """Agente de Reinforcement Learning usando PPO"""

    def __init__(self, environment: PricingEnvironment):
        self.env = environment
        self.model: Optional[PPO] = None
        self.is_trained = False

    def build_model(self):
        """Construye el modelo PPO"""
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )
        print("✓ Modelo PPO construido")

    def train(self, timesteps: int):
        """Entrena el agente"""
        if self.model is None:
            self.build_model()

        print(f"Entrenando agente PPO para {self.env.product_data.category}...")
        self.model.learn(total_timesteps=timesteps, progress_bar=False)
        self.is_trained = True
        print(f"✓ Agente entrenado con {timesteps} timesteps")

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Predice acción óptima"""
        if self.model is None or not self.is_trained:
            return np.array([0.0])  # Sin cambio de precio

        action, _ = self.model.predict(observation, deterministic=True)
        return action

    def evaluate(self, episodes: int = 10) -> Tuple[float, float]:
        """Evalúa el agente entrenado"""
        total_rewards = []
        total_revenues = []

        for _ in range(episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_revenue = 0
            done = False

            while not done:
                action = self.predict(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_revenue += info['revenue']
                done = terminated or truncated

            total_rewards.append(episode_reward)
            total_revenues.append(episode_revenue)

        return np.mean(total_rewards), np.mean(total_revenues)


# ======================= VISUALIZER =======================
class PricingVisualizer:
    """Visualizador Pygame para el sistema de pricing dinámico"""

    def __init__(self, products: Dict[str, ProductData]):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Task C: Dynamic Pricing with RL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)

        self.products = products
        self.current_product = list(products.keys())[0]

        # Ambientes y agentes
        self.environments: Dict[str, PricingEnvironment] = {}
        self.agents: Dict[str, RLAgent] = {}
        self.static_revenues: Dict[str, List[float]] = {}

        # Estado de simulación
        self.is_running = False
        self.current_episode = 0
        self.max_episodes = 100
        self.revenue_history: List[float] = []
        self.static_revenue_history: List[float] = []
        self.price_history: List[float] = []

        # UI Elements
        self.initial_price = products[self.current_product].base_price
        self.price_slider_rect = pygame.Rect(50, 680, 300, 20)
        self.price_slider_value = 1.0  # Multiplicador

        self.reset_button_rect = pygame.Rect(400, 670, 150, 40)
        self.train_button_rect = pygame.Rect(570, 670, 150, 40)
        self.toggle_button_rect = pygame.Rect(740, 670, 200, 40)

        # Selector de producto
        self.product_buttons = {}
        x_start = 50
        for i, product in enumerate(products.keys()):
            self.product_buttons[product] = pygame.Rect(x_start + i * 150, 730, 140, 35)

        # Inicializar ambientes
        self._initialize_environments()

    def _initialize_environments(self):
        """Inicializa ambientes y agentes para cada producto"""
        for product_name, product_data in self.products.items():
            env = PricingEnvironment(product_data, max_steps=self.max_episodes)
            agent = RLAgent(env)

            self.environments[product_name] = env
            self.agents[product_name] = agent
            self.static_revenues[product_name] = []

        print("✓ Ambientes inicializados")

    def _train_current_agent(self):
        """Entrena el agente del producto actual"""
        agent = self.agents[self.current_product]
        agent.train(TRAINING_TIMESTEPS)

    def _reset_simulation(self):
        """Reinicia la simulación"""
        self.is_running = False
        self.current_episode = 0
        self.revenue_history.clear()
        self.static_revenue_history.clear()
        self.price_history.clear()

        # Reiniciar ambiente
        env = self.environments[self.current_product]
        env.base_price = self.products[self.current_product].base_price * self.price_slider_value
        env.current_price = env.base_price
        env.reset()

        print(f"✓ Simulación reiniciada - Precio base: ${env.base_price:.2f}")

    def _step_simulation(self):
        """Ejecuta un paso de la simulación"""
        if not self.is_running or self.current_episode >= self.max_episodes:
            return

        env = self.environments[self.current_product]
        agent = self.agents[self.current_product]

        # Obtener observación actual
        obs = env._get_observation()

        # Predicción del agente RL
        action = agent.predict(obs)
        _, _, terminated, truncated, info = env.step(action)

        # Guardar métricas del RL
        self.revenue_history.append(info['revenue'])
        self.price_history.append(info['price'])

        # ===== COMPARACIÓN JUSTA CON PRECIO ESTÁTICO =====
        # Simular demanda independiente con precio base (no usar cantidad del RL)
        static_demand = env._simulate_demand(env.base_price)
        static_quantity = max(1, int(np.random.poisson(env.product_data.avg_quantity * static_demand)))
        static_revenue = env.base_price * static_quantity
        self.static_revenue_history.append(static_revenue)

        self.current_episode += 1

        if terminated or truncated:
            self.is_running = False
            total_rl = sum(self.revenue_history)
            total_static = sum(self.static_revenue_history)
            improvement = ((total_rl - total_static) / total_static) * 100 if total_static > 0 else 0
            print(f"✓ Episodio completado - RL: ${total_rl:.2f}, Estático: ${total_static:.2f}, Mejora: {improvement:+.2f}%")

    def _draw_graph(self):
        """Dibuja gráfica de ingresos en tiempo real"""
        graph_rect = pygame.Rect(50, 50, 1100, 550)
        pygame.draw.rect(self.screen, WHITE, graph_rect)
        pygame.draw.rect(self.screen, BLACK, graph_rect, 2)

        if len(self.revenue_history) < 2:
            # Mensaje cuando no hay datos
            agent = self.agents[self.current_product]
            if not agent.is_trained:
                text1 = self.font.render("Paso 1: Haz clic en 'Train Agent' para entrenar el modelo RL", True, DARK_GRAY)
                text2 = self.font.render("(Esto puede tomar 10-20 segundos)", True, DARK_GRAY)
                text3 = self.small_font.render("El agente aprenderá a ajustar precios dinámicamente para maximizar ingresos", True, DARK_GRAY)
            else:
                text1 = self.font.render("Paso 2: Haz clic en 'Start/Stop' para comenzar la simulación", True, GREEN)
                text2 = self.font.render("Observa cómo el RL ajusta precios vs. precio estático", True, GREEN)
                text3 = self.small_font.render("Puedes ajustar el precio inicial con el slider antes de iniciar", True, DARK_GRAY)

            text1_rect = text1.get_rect(center=(graph_rect.centerx, graph_rect.centery - 40))
            text2_rect = text2.get_rect(center=(graph_rect.centerx, graph_rect.centery))
            text3_rect = text3.get_rect(center=(graph_rect.centerx, graph_rect.centery + 40))

            self.screen.blit(text1, text1_rect)
            self.screen.blit(text2, text2_rect)
            self.screen.blit(text3, text3_rect)
            return

        # Crear gráfica con matplotlib
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), dpi=100)

        # Gráfica 1: Revenue Comparison
        episodes = list(range(len(self.revenue_history)))
        ax1.plot(episodes, self.revenue_history, label='RL Dynamic Pricing', color='blue', linewidth=2)
        ax1.plot(episodes, self.static_revenue_history, label='Static Pricing', color='red', linewidth=2, linestyle='--')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Revenue ($)')
        ax1.set_title(f'Revenue Comparison - {self.current_product}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Gráfica 2: Price Evolution
        ax2.plot(episodes, self.price_history, label='Dynamic Price', color='green', linewidth=2)
        base_price = self.environments[self.current_product].base_price
        ax2.axhline(y=base_price, color='orange', linestyle='--', label='Base Price', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Price ($)')
        ax2.set_title('Price Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Convertir a superficie Pygame (método actualizado)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Método actualizado para versiones nuevas de matplotlib
        buf = canvas.buffer_rgba()
        size = canvas.get_width_height()

        surf = pygame.image.frombuffer(buf, size, "RGBA")
        self.screen.blit(surf, (graph_rect.x, graph_rect.y))

        plt.close(fig)

    def _draw_ui(self):
        """Dibuja elementos de UI"""
        # Título
        title = self.font.render("Dynamic Pricing with Reinforcement Learning (PPO)", True, BLACK)
        self.screen.blit(title, (50, 10))

        # Price Slider
        slider_label = self.small_font.render(
            f"Initial Price Multiplier: {self.price_slider_value:.2f}x (${self.products[self.current_product].base_price * self.price_slider_value:.2f})",
            True, BLACK
        )
        self.screen.blit(slider_label, (50, 655))

        # Slider track
        pygame.draw.rect(self.screen, GRAY, self.price_slider_rect)
        # Slider handle
        handle_x = self.price_slider_rect.x + int((self.price_slider_value - 0.5) / 1.5 * self.price_slider_rect.width)
        handle_x = max(self.price_slider_rect.x, min(handle_x, self.price_slider_rect.right))
        pygame.draw.circle(self.screen, BLUE, (handle_x, self.price_slider_rect.centery), 12)

        # Reset Button
        pygame.draw.rect(self.screen, ORANGE, self.reset_button_rect, border_radius=5)
        reset_text = self.small_font.render("Reset", True, WHITE)
        reset_rect = reset_text.get_rect(center=self.reset_button_rect.center)
        self.screen.blit(reset_text, reset_rect)

        # Train Button
        agent = self.agents[self.current_product]
        train_color = GREEN if agent.is_trained else GRAY
        pygame.draw.rect(self.screen, train_color, self.train_button_rect, border_radius=5)
        train_text = self.small_font.render("Train Agent", True, WHITE)
        train_rect = train_text.get_rect(center=self.train_button_rect.center)
        self.screen.blit(train_text, train_rect)

        # Toggle Button
        toggle_color = GREEN if self.is_running else RED
        pygame.draw.rect(self.screen, toggle_color, self.toggle_button_rect, border_radius=5)
        toggle_text = self.small_font.render("Start/Stop", True, WHITE)
        toggle_rect = toggle_text.get_rect(center=self.toggle_button_rect.center)
        self.screen.blit(toggle_text, toggle_rect)

        # Product Selection Buttons
        for product, rect in self.product_buttons.items():
            color = BLUE if product == self.current_product else LIGHT_BLUE
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            pygame.draw.rect(self.screen, BLACK, rect, 2, border_radius=5)
            product_text = self.small_font.render(product, True, BLACK)
            product_rect = product_text.get_rect(center=rect.center)
            self.screen.blit(product_text, product_rect)

        # Stats
        if len(self.revenue_history) > 0:
            total_rl = sum(self.revenue_history)
            total_static = sum(self.static_revenue_history)
            improvement = ((total_rl - total_static) / total_static) * 100 if total_static > 0 else 0

            stats_y = 620
            stats = [
                f"Episode: {self.current_episode}/{self.max_episodes}",
                f"Total RL Revenue: ${total_rl:.2f}",
                f"Total Static Revenue: ${total_static:.2f}",
                f"Improvement: {improvement:.2f}%"
            ]

            for i, stat in enumerate(stats):
                color = GREEN if improvement > 0 and i == 3 else BLACK
                stat_text = self.small_font.render(stat, True, color)
                self.screen.blit(stat_text, (970 - i * 0, stats_y + i * 25))

    def _handle_events(self) -> bool:
        """Maneja eventos de Pygame"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos

                # Reset button
                if self.reset_button_rect.collidepoint(mouse_pos):
                    self._reset_simulation()

                # Train button
                elif self.train_button_rect.collidepoint(mouse_pos):
                    if not self.agents[self.current_product].is_trained:
                        self._train_current_agent()

                # Toggle button
                elif self.toggle_button_rect.collidepoint(mouse_pos):
                    if not self.is_running and self.current_episode < self.max_episodes:
                        if not self.agents[self.current_product].is_trained:
                            print("⚠ Por favor entrena el agente primero")
                        else:
                            self.is_running = True
                    else:
                        self.is_running = False

                # Product buttons
                for product, rect in self.product_buttons.items():
                    if rect.collidepoint(mouse_pos):
                        self.current_product = product
                        self._reset_simulation()

                # Slider
                if self.price_slider_rect.collidepoint(mouse_pos):
                    self._update_slider(mouse_pos[0])

            elif event.type == pygame.MOUSEMOTION:
                if event.buttons[0]:  # Left button held
                    if self.price_slider_rect.collidepoint(event.pos):
                        self._update_slider(event.pos[0])

        return True

    def _update_slider(self, mouse_x: int):
        """Actualiza el valor del slider"""
        relative_x = mouse_x - self.price_slider_rect.x
        ratio = relative_x / self.price_slider_rect.width
        self.price_slider_value = 0.5 + ratio * 1.5  # Range: 0.5 to 2.0
        self.price_slider_value = max(0.5, min(2.0, self.price_slider_value))

    def run(self):
        """Loop principal de la aplicación"""
        print("\n=== Aplicación en ejecución ===\n")
        running = True

        while running:
            running = self._handle_events()

            # Step simulation
            if self.is_running:
                self._step_simulation()

            # Render
            self.screen.fill(WHITE)
            self._draw_graph()
            self._draw_ui()

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()


# ======================= MAIN APPLICATION =======================
class DynamicPricingApp:
    """Aplicación principal de pricing dinámico"""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.data_loader = DataLoader(csv_path)
        self.visualizer: Optional[PricingVisualizer] = None

    def initialize(self) -> bool:
        """Inicializa la aplicación"""
        print("=== Inicializando Dynamic Pricing System ===")

        # Cargar datos
        if not self.data_loader.load_data():
            return False

        # Procesar productos
        products = self.data_loader.process_products()
        if not products:
            print("✗ No se encontraron productos")
            return False

        # Inicializar visualizador
        self.visualizer = PricingVisualizer(products)
        print("✓ Aplicación inicializada correctamente\n")

        return True

    def run(self):
        """Ejecuta la aplicación"""
        if self.visualizer is None:
            print("✗ Aplicación no inicializada")
            return

        self.visualizer.run()


# ======================= ENTRY POINT =======================
def main():
    """Punto de entrada principal"""
    csv_path = "retail_sales_dataset.csv"

    app = DynamicPricingApp(csv_path)

    if app.initialize():
        app.run()
    else:
        print("✗ Error al inicializar la aplicación")


if __name__ == "__main__":
    main()
