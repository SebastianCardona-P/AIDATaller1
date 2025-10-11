"""
Task B: Demand Forecasting (Time-Series with Small LSTM)
Predicción de ventas futuras para respaldar decisiones de inventario
Siguiendo principios SOLID, KISS y clean code
"""

import pygame
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import os
import io

# TensorFlow/Keras imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reducir logs de TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ======================= CONSTANTS =======================
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800
PLOT_WIDTH = 1000
PLOT_HEIGHT = 600
PLOT_OFFSET_X = 50
PLOT_OFFSET_Y = 50

# Model parameters
SEQUENCE_LENGTH = 10  # Ajustado: balance entre contexto y overfitting
LSTM_UNITS = 128  # Optimizado
LSTM_LAYERS = 3
DROPOUT_RATE = 0.35  # Ajustado para mejor balance
EPOCHS_MIN = 100
EPOCHS_MAX = 500
EPOCHS_STEP = 50
PREDICTION_WEEKS = 4
BATCH_SIZE = 4  # Más pequeño para aprendizaje más fino

# Colors
COLOR_BG = (245, 245, 245)
COLOR_PANEL = (255, 255, 255)
COLOR_TEXT = (40, 40, 40)
COLOR_SLIDER_BG = (200, 200, 200)
COLOR_SLIDER_FG = (70, 130, 180)
COLOR_BUTTON = (70, 130, 180)
COLOR_BUTTON_HOVER = (100, 149, 237)
COLOR_BUTTON_TEXT = (255, 255, 255)


# ======================= DATA CLASSES =======================
@dataclass
class ForecastMetrics:
    """Métricas de evaluación del modelo"""
    mae: float
    rmse: float
    category: str
    training_epochs: int


@dataclass
class TimeSeriesData:
    """Datos de serie temporal procesados"""
    dates: List[datetime]
    values: np.ndarray
    category: str


# ======================= DATA LOADER =======================
class SalesDataLoader:
    """Responsable de cargar y preprocesar datos de ventas"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        self.categories = []
        self._load_data()

    def _load_data(self):
        """Carga los datos desde el CSV"""
        try:
            self.df = pd.read_csv(self.filepath)
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.categories = sorted(self.df['Product Category'].unique().tolist())
            print(f"✓ Datos cargados: {len(self.df)} transacciones")
            print(f"✓ Categorías encontradas: {self.categories}")
        except Exception as e:
            print(f"Error al cargar datos: {e}")
            raise

    def get_weekly_sales(self, category: str) -> TimeSeriesData:
        """Obtiene ventas semanales agregadas por categoría"""
        if self.df is None:
            raise ValueError("Datos no cargados")

        # Filtrar por categoría
        category_df = self.df[self.df['Product Category'] == category].copy()

        # Agrupar por semana
        category_df['Week'] = category_df['Date'].dt.to_period('W').dt.start_time
        weekly_sales = category_df.groupby('Week').agg({
            'Total Amount': 'sum',
            'Quantity': 'sum'
        }).reset_index()

        # Rellenar semanas faltantes con ceros
        date_range = pd.date_range(
            start=weekly_sales['Week'].min(),
            end=weekly_sales['Week'].max(),
            freq='W-MON'
        )

        full_weekly = pd.DataFrame({'Week': date_range})
        full_weekly = full_weekly.merge(weekly_sales, on='Week', how='left')
        full_weekly = full_weekly.fillna(0)

        # SUAVIZADO: Aplicar media móvil para reducir volatilidad
        window_size = 3  # Ventana de 3 semanas
        full_weekly['Total Amount'] = full_weekly['Total Amount'].rolling(
            window=window_size, min_periods=1, center=True
        ).mean()

        return TimeSeriesData(
            dates=full_weekly['Week'].tolist(),
            values=full_weekly['Total Amount'].values,
            category=category
        )

    def get_categories(self) -> List[str]:
        """Retorna la lista de categorías disponibles"""
        return self.categories


# ======================= TIME SERIES PROCESSOR =======================
class TimeSeriesProcessor:
    """Procesa datos de series temporales para LSTM"""

    def __init__(self, sequence_length: int = SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepara secuencias para entrenamiento del LSTM
        Retorna: (X_train, y_train, scaled_data)
        """
        # Escalar datos
        data_reshaped = data.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data_reshaped)

        X, y = [], []

        # Crear secuencias
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i + self.sequence_length])
            y.append(scaled_data[i + self.sequence_length])

        X = np.array(X)
        y = np.array(y)

        return X, y, scaled_data

    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """Revierte la normalización"""
        return self.scaler.inverse_transform(scaled_data)


# ======================= LSTM MODEL =======================
class LSTMForecaster:
    """Modelo LSTM para forecasting de demanda"""

    def __init__(self, sequence_length: int = SEQUENCE_LENGTH,
                 lstm_units: int = LSTM_UNITS,
                 dropout_rate: float = DROPOUT_RATE):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.processor = TimeSeriesProcessor(sequence_length)
        self.history = None

    def build_model(self):
        """Construye la arquitectura del modelo LSTM"""
        self.model = Sequential()

        # Capas LSTM
        for i in range(LSTM_LAYERS):
            return_sequences = (i < LSTM_LAYERS - 1)  # Todas menos la última
            self.model.add(LSTM(self.lstm_units, return_sequences=return_sequences,
                                 input_shape=(self.sequence_length, 1)))
            self.model.add(Dropout(self.dropout_rate))

        # Capa de salida
        self.model.add(Dense(1))

        self.model.compile(
            optimizer=Adam(learning_rate=0.002),
            loss='mse',
            metrics=['mae']
        )

        print("✓ Modelo LSTM construido")
        return self.model

    def train(self, data: np.ndarray, epochs: int = 50, verbose: int = 0) -> Dict:
        """Entrena el modelo con los datos"""
        if self.model is None:
            self.build_model()

        # Preparar datos
        X_train, y_train, scaled_data = self.processor.prepare_sequences(data)

        if len(X_train) < 5:
            raise ValueError("No hay suficientes datos para entrenar (mínimo 5 secuencias)")

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min', restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=0, mode='min', min_delta=1e-4)

        # Entrenar
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=min(BATCH_SIZE, len(X_train)),
            verbose=verbose,
            validation_split=0.2 if len(X_train) > 10 else 0.0,
            callbacks=[early_stopping, reduce_lr]
        )

        return {
            'loss': self.history.history['loss'][-1],
            'mae': self.history.history['mae'][-1]
        }

    def predict_future(self, data: np.ndarray, n_steps: int = PREDICTION_WEEKS) -> np.ndarray:
        """Predice n_steps pasos futuros"""
        if self.model is None:
            raise ValueError("Modelo no entrenado")

        # Preparar datos iniciales
        _, _, scaled_data = self.processor.prepare_sequences(data)

        # Usar las últimas sequence_length observaciones
        last_sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)

        predictions = []
        current_sequence = last_sequence.copy()

        # Predicción iterativa
        for _ in range(n_steps):
            # Predecir siguiente valor
            next_pred = self.model.predict(current_sequence, verbose=0)
            predictions.append(next_pred[0, 0])

            # Actualizar secuencia
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred[0, 0]

        # Revertir normalización
        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_original = self.processor.inverse_transform(predictions_array)

        return predictions_original.flatten()

    def evaluate(self, data: np.ndarray) -> ForecastMetrics:
        """Evalúa el modelo con métricas MAE y RMSE"""
        if self.model is None:
            raise ValueError("Modelo no entrenado")

        X_test, y_test, _ = self.processor.prepare_sequences(data)

        # Hacer predicciones
        predictions = self.model.predict(X_test, verbose=0)

        # Revertir normalización
        y_test_original = self.processor.inverse_transform(y_test)
        predictions_original = self.processor.inverse_transform(predictions)

        # Calcular métricas
        mae = mean_absolute_error(y_test_original, predictions_original)
        rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))

        return mae, rmse


# ======================= FORECAST MANAGER =======================
class ForecastManager:
    """Gestiona el forecasting para múltiples categorías"""

    def __init__(self, data_loader: SalesDataLoader):
        self.data_loader = data_loader
        self.forecasters: Dict[str, LSTMForecaster] = {}
        self.time_series: Dict[str, TimeSeriesData] = {}
        self.predictions: Dict[str, np.ndarray] = {}
        self.metrics: Dict[str, ForecastMetrics] = {}

    def prepare_category(self, category: str):
        """Prepara datos de serie temporal para una categoría"""
        self.time_series[category] = self.data_loader.get_weekly_sales(category)
        print(f"✓ Serie temporal preparada para {category}: {len(self.time_series[category].values)} semanas")

    def train_category(self, category: str, epochs: int = 50):
        """Entrena modelo LSTM para una categoría"""
        if category not in self.time_series:
            self.prepare_category(category)

        ts_data = self.time_series[category]

        # Crear y entrenar modelo
        forecaster = LSTMForecaster()
        self.forecasters[category] = forecaster

        print(f"Entrenando modelo para {category} con {epochs} épocas...")
        forecaster.train(ts_data.values, epochs=epochs, verbose=0)

        # Hacer predicciones
        self.predictions[category] = forecaster.predict_future(ts_data.values)

        # Evaluar
        mae, rmse = forecaster.evaluate(ts_data.values)
        self.metrics[category] = ForecastMetrics(
            mae=mae,
            rmse=rmse,
            category=category,
            training_epochs=epochs
        )

        print(f"✓ Modelo entrenado - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    def export_predictions(self, category: str, filepath: str = "predictions.csv"):
        """Exporta predicciones a CSV"""
        if category not in self.predictions:
            raise ValueError(f"No hay predicciones para {category}")

        ts_data = self.time_series[category]
        pred_values = self.predictions[category]

        # Crear fechas futuras
        last_date = ts_data.dates[-1]
        future_dates = [last_date + timedelta(weeks=i+1) for i in range(len(pred_values))]

        # Crear DataFrame
        export_df = pd.DataFrame({
            'Date': future_dates,
            'Category': category,
            'Predicted_Sales': pred_values,
            'MAE': self.metrics[category].mae,
            'RMSE': self.metrics[category].rmse
        })

        export_df.to_csv(filepath, index=False)
        print(f"✓ Predicciones exportadas a {filepath}")


# ======================= PLOT GENERATOR =======================
class PlotGenerator:
    """Genera gráficos de forecasting"""

    @staticmethod
    def create_forecast_plot(ts_data: TimeSeriesData,
                            predictions: np.ndarray,
                            metrics: ForecastMetrics,
                            width: int = 10,
                            height: int = 6) -> io.BytesIO:
        """Crea un gráfico de forecast y lo retorna como BytesIO"""
        fig, ax = plt.subplots(figsize=(width, height))

        # Datos históricos
        ax.plot(ts_data.dates, ts_data.values,
               label='Historical Sales', color='#2E86AB', linewidth=2, marker='o')

        # Predicciones
        last_date = ts_data.dates[-1]
        future_dates = [last_date + timedelta(weeks=i+1) for i in range(len(predictions))]

        ax.plot(future_dates, predictions,
               label='Forecast', color='#A23B72', linewidth=2,
               marker='s', linestyle='--')

        # Línea de conexión
        ax.plot([ts_data.dates[-1], future_dates[0]],
               [ts_data.values[-1], predictions[0]],
               color='gray', linestyle=':', linewidth=1)

        # Configuración
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sales ($)', fontsize=12, fontweight='bold')
        ax.set_title(f'Sales Forecast - {ts_data.category}\nMAE: {metrics.mae:.2f} | RMSE: {metrics.rmse:.2f}',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Rotar etiquetas de fecha
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Convertir a BytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        return buf


# ======================= UI COMPONENTS =======================
class Slider:
    """Control deslizante para UI"""

    def __init__(self, x: int, y: int, width: int, height: int,
                 min_val: int, max_val: int, initial_val: int, step: int, label: str):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.step = step
        self.label = label
        self.dragging = False
        self.handle_radius = height // 2 + 2

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Maneja eventos del mouse"""
        mouse_pos = pygame.mouse.get_pos()
        handle_x = self._value_to_x(self.value)
        handle_rect = pygame.Rect(
            handle_x - self.handle_radius,
            self.rect.y - 2,
            self.handle_radius * 2,
            self.rect.height + 4
        )

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if handle_rect.collidepoint(mouse_pos):
                    self.dragging = True
                    return True
                elif self.rect.collidepoint(mouse_pos):
                    # Click en la barra
                    new_val = self._x_to_value(mouse_pos[0])
                    if new_val != self.value:
                        self.value = new_val
                        return True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.dragging:
                self.dragging = False
                return True

        elif event.type == pygame.MOUSEMOTION and self.dragging:
            new_val = self._x_to_value(mouse_pos[0])
            if new_val != self.value:
                self.value = new_val
                return True

        return False

    def _value_to_x(self, value: int) -> int:
        """Convierte valor a posición X"""
        ratio = (value - self.min_val) / (self.max_val - self.min_val)
        return int(self.rect.x + ratio * self.rect.width)

    def _x_to_value(self, x: int) -> int:
        """Convierte posición X a valor"""
        x = max(self.rect.x, min(x, self.rect.x + self.rect.width))
        ratio = (x - self.rect.x) / self.rect.width
        value = self.min_val + ratio * (self.max_val - self.min_val)
        # Ajustar al step más cercano
        value = round(value / self.step) * self.step
        return int(max(self.min_val, min(value, self.max_val)))

    def draw(self, surface: pygame.Surface, font: pygame.font.Font):
        """Dibuja el slider"""
        # Label
        label_surface = font.render(f"{self.label}: {self.value}", True, COLOR_TEXT)
        surface.blit(label_surface, (self.rect.x, self.rect.y - 25))

        # Barra de fondo
        pygame.draw.rect(surface, COLOR_SLIDER_BG, self.rect, border_radius=self.rect.height // 2)

        # Barra de progreso
        handle_x = self._value_to_x(self.value)
        progress_rect = pygame.Rect(self.rect.x, self.rect.y,
                                    handle_x - self.rect.x, self.rect.height)
        pygame.draw.rect(surface, COLOR_SLIDER_FG, progress_rect, border_radius=self.rect.height // 2)

        # Handle
        pygame.draw.circle(surface, COLOR_SLIDER_FG, (handle_x, self.rect.centery), self.handle_radius)
        pygame.draw.circle(surface, (255, 255, 255), (handle_x, self.rect.centery), self.handle_radius - 3)


class Button:
    """Botón clickeable"""

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

        text_surface = font.render(self.text, True, COLOR_BUTTON_TEXT)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)


class CategorySelector:
    """Selector de categorías"""

    def __init__(self, x: int, y: int, width: int, height: int, categories: List[str]):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.categories = categories
        self.selected_index = 0

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Maneja eventos del mouse"""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = pygame.mouse.get_pos()

            for i, category in enumerate(self.categories):
                button_y = self.y + i * (self.height + 5)
                button_rect = pygame.Rect(self.x, button_y, self.width, self.height)

                if button_rect.collidepoint(mouse_pos):
                    if self.selected_index != i:
                        self.selected_index = i
                        return True
        return False

    def get_selected_category(self) -> str:
        """Retorna la categoría seleccionada"""
        return self.categories[self.selected_index]

    def draw(self, surface: pygame.Surface, font: pygame.font.Font):
        """Dibuja el selector"""
        mouse_pos = pygame.mouse.get_pos()

        for i, category in enumerate(self.categories):
            button_y = self.y + i * (self.height + 5)
            button_rect = pygame.Rect(self.x, button_y, self.width, self.height)

            # Determinar color
            if i == self.selected_index:
                color = COLOR_SLIDER_FG
                text_color = COLOR_BUTTON_TEXT
            elif button_rect.collidepoint(mouse_pos):
                color = COLOR_BUTTON_HOVER
                text_color = COLOR_BUTTON_TEXT
            else:
                color = COLOR_PANEL
                text_color = COLOR_TEXT

            # Dibujar botón
            pygame.draw.rect(surface, color, button_rect, border_radius=5)
            pygame.draw.rect(surface, COLOR_SLIDER_BG, button_rect, 2, border_radius=5)

            # Dibujar texto
            text_surface = font.render(category, True, text_color)
            text_rect = text_surface.get_rect(center=button_rect.center)
            surface.blit(text_surface, text_rect)


# ======================= FORECAST APPLICATION =======================
class ForecastApplication:
    """Aplicación principal de forecasting con Pygame"""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Task B: Demand Forecasting with LSTM")
        self.clock = pygame.time.Clock()
        self.running = True

        # Fonts
        self.font_title = pygame.font.Font(None, 32)
        self.font_normal = pygame.font.Font(None, 20)
        self.font_small = pygame.font.Font(None, 16)

        # Cargar datos
        print("\n=== Inicializando Demand Forecasting System ===")
        self.data_loader = SalesDataLoader("retail_sales_dataset.csv")
        self.forecast_manager = ForecastManager(self.data_loader)

        # Preparar categorías
        categories = self.data_loader.get_categories()[:5]  # Limitar a 5 categorías
        for category in categories:
            self.forecast_manager.prepare_category(category)

        # UI Components
        panel_x = PLOT_OFFSET_X + PLOT_WIDTH + 30

        self.category_selector = CategorySelector(
            panel_x, PLOT_OFFSET_Y + 50, 250, 35, categories
        )

        self.epochs_slider = Slider(
            panel_x, PLOT_OFFSET_Y + 300, 250, 20,
            EPOCHS_MIN, EPOCHS_MAX, 50, EPOCHS_STEP, "Training Epochs"
        )

        self.train_button = Button(
            panel_x, PLOT_OFFSET_Y + 370, 250, 40, "Train Model"
        )

        self.export_button = Button(
            panel_x, PLOT_OFFSET_Y + 420, 250, 40, "Export Predictions"
        )

        # Estado
        self.current_plot_surface = None
        self.is_training = False
        self.status_message = "Select category and click Train Model"
        self.status_color = COLOR_TEXT

        print("✓ Aplicación inicializada correctamente\n")

    def train_current_category(self):
        """Entrena el modelo para la categoría seleccionada"""
        category = self.category_selector.get_selected_category()
        epochs = self.epochs_slider.value

        self.is_training = True
        self.status_message = f"Training {category}..."
        self.status_color = (255, 165, 0)  # Orange

        try:
            # Entrenar modelo
            self.forecast_manager.train_category(category, epochs=epochs)

            # Generar plot
            ts_data = self.forecast_manager.time_series[category]
            predictions = self.forecast_manager.predictions[category]
            metrics = self.forecast_manager.metrics[category]

            plot_buf = PlotGenerator.create_forecast_plot(ts_data, predictions, metrics)

            # Convertir a superficie de Pygame
            plot_image = pygame.image.load(plot_buf, 'png')
            self.current_plot_surface = pygame.transform.scale(plot_image, (PLOT_WIDTH, PLOT_HEIGHT))

            self.status_message = f"✓ Model trained - MAE: {metrics.mae:.2f}, RMSE: {metrics.rmse:.2f}"
            self.status_color = (0, 128, 0)  # Green

        except Exception as e:
            self.status_message = f"✗ Error: {str(e)}"
            self.status_color = (255, 0, 0)  # Red
            print(f"Error durante entrenamiento: {e}")

        self.is_training = False

    def export_current_predictions(self):
        """Exporta las predicciones actuales a CSV"""
        category = self.category_selector.get_selected_category()

        if category not in self.forecast_manager.predictions:
            self.status_message = "✗ Train model first before exporting"
            self.status_color = (255, 0, 0)
            return

        try:
            filename = f"predictions_{category.lower()}.csv"
            self.forecast_manager.export_predictions(category, filename)
            self.status_message = f"✓ Exported to {filename}"
            self.status_color = (0, 128, 0)
        except Exception as e:
            self.status_message = f"✗ Export error: {str(e)}"
            self.status_color = (255, 0, 0)

    def handle_events(self):
        """Maneja eventos de Pygame"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            # UI events
            if not self.is_training:
                if self.category_selector.handle_event(event):
                    # Categoría cambiada, actualizar plot si existe
                    category = self.category_selector.get_selected_category()
                    if category in self.forecast_manager.predictions:
                        # Regenerar plot
                        ts_data = self.forecast_manager.time_series[category]
                        predictions = self.forecast_manager.predictions[category]
                        metrics = self.forecast_manager.metrics[category]
                        plot_buf = PlotGenerator.create_forecast_plot(ts_data, predictions, metrics)
                        plot_image = pygame.image.load(plot_buf, 'png')
                        self.current_plot_surface = pygame.transform.scale(plot_image, (PLOT_WIDTH, PLOT_HEIGHT))

                self.epochs_slider.handle_event(event)

                if self.train_button.handle_event(event):
                    self.train_current_category()

                if self.export_button.handle_event(event):
                    self.export_current_predictions()

    def render(self):
        """Renderiza la aplicación"""
        self.screen.fill(COLOR_BG)

        # Plot area
        plot_rect = pygame.Rect(PLOT_OFFSET_X, PLOT_OFFSET_Y, PLOT_WIDTH, PLOT_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_PANEL, plot_rect)
        pygame.draw.rect(self.screen, COLOR_SLIDER_BG, plot_rect, 2)

        if self.current_plot_surface:
            self.screen.blit(self.current_plot_surface, (PLOT_OFFSET_X, PLOT_OFFSET_Y))
        else:
            # Mensaje de bienvenida
            welcome_text = self.font_normal.render("Select a category and train the model", True, COLOR_TEXT)
            welcome_rect = welcome_text.get_rect(center=plot_rect.center)
            self.screen.blit(welcome_text, welcome_rect)

        # Control panel
        panel_x = PLOT_OFFSET_X + PLOT_WIDTH + 30
        panel_rect = pygame.Rect(panel_x - 10, PLOT_OFFSET_Y - 10, 270, 520)
        pygame.draw.rect(self.screen, COLOR_PANEL, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, COLOR_SLIDER_BG, panel_rect, 2, border_radius=10)

        # Title
        title_text = self.font_title.render("Controls", True, COLOR_TEXT)
        self.screen.blit(title_text, (panel_x, PLOT_OFFSET_Y))

        # Category selector
        category_label = self.font_normal.render("Product Category:", True, COLOR_TEXT)
        self.screen.blit(category_label, (panel_x, PLOT_OFFSET_Y + 40))
        self.category_selector.draw(self.screen, self.font_small)

        # Epochs slider
        self.epochs_slider.draw(self.screen, self.font_normal)

        # Buttons
        self.train_button.draw(self.screen, self.font_normal)
        self.export_button.draw(self.screen, self.font_normal)

        # Status message
        status_y = PLOT_OFFSET_Y + PLOT_HEIGHT + 20
        status_text = self.font_small.render(self.status_message, True, self.status_color)
        self.screen.blit(status_text, (PLOT_OFFSET_X, status_y))

        # Info
        info_lines = [
            "LSTM Demand Forecasting System",
            f"Sequence Length: {SEQUENCE_LENGTH} weeks",
            f"Prediction Horizon: {PREDICTION_WEEKS} weeks",
            "Metrics: MAE (Mean Absolute Error), RMSE (Root Mean Square Error)"
        ]

        info_y = status_y + 30
        for i, line in enumerate(info_lines):
            info_text = self.font_small.render(line, True, COLOR_TEXT)
            self.screen.blit(info_text, (PLOT_OFFSET_X, info_y + i * 20))

        pygame.display.flip()

    def run(self):
        """Loop principal de la aplicación"""
        print("=== Aplicación en ejecución ===\n")

        while self.running:
            self.handle_events()
            self.render()
            self.clock.tick(60)

        pygame.quit()
        print("\n=== Aplicación cerrada ===")


# ======================= MAIN =======================
def main():
    """Punto de entrada principal"""
    try:
        app = ForecastApplication()
        app.run()
    except Exception as e:
        print(f"\n✗ Error fatal: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
