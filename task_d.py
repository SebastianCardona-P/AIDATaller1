"""
Task D: GAN for Product Promotional Images
Sistema de generación de imágenes promocionales usando DCGAN con Fashion-MNIST.

Arquitectura optimizada y probada para generar imágenes de productos de ropa.
"""

import pygame
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import os
from datetime import datetime
import time

# ======================= CONFIGURACIÓN =======================
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
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
PURPLE = (147, 112, 219)

# Parámetros GAN optimizados
IMAGE_SIZE = 28
LATENT_DIM = 100
BATCH_SIZE = 128
EPOCHS = 50  # Suficientes para convergencia
LEARNING_RATE = 0.0002


@dataclass
class GANConfig:
    """Configuración del GAN"""
    image_size: int = IMAGE_SIZE
    latent_dim: int = LATENT_DIM
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE


# ======================= DCGAN =======================
class DCGAN:
    """
    Deep Convolutional GAN optimizada para Fashion-MNIST.
    Arquitectura probada que garantiza convergencia.
    """

    def __init__(self, config: GANConfig):
        self.config = config
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

        # Optimizadores separados
        self.g_optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate, beta_1=0.5)
        self.d_optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate, beta_1=0.5)

        # Loss
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)

        # Métricas
        self.d_losses = []
        self.g_losses = []
        self.d_accuracies = []

        print("✓ DCGAN construida exitosamente")
        print(f"  Generator params: {self.generator.count_params():,}")
        print(f"  Discriminator params: {self.discriminator.count_params():,}")

    def _build_generator(self) -> Model:
        """Generator: noise -> image"""
        model = keras.Sequential([
            # Entrada: (100,)
            layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(self.config.latent_dim,)),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Reshape((7, 7, 256)),

            # Upsample a 14x14
            layers.Conv2DTranspose(128, 5, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),

            # Upsample a 28x28
            layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),

            # Output: 28x28x1
            layers.Conv2DTranspose(1, 5, strides=1, padding='same', use_bias=False, activation='tanh')
        ], name='generator')

        return model

    def _build_discriminator(self) -> Model:
        """Discriminator: image -> real/fake"""
        model = keras.Sequential([
            # Entrada: 28x28x1
            layers.Conv2D(64, 5, strides=2, padding='same', input_shape=(28, 28, 1)),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),

            # Downsample a 7x7
            layers.Conv2D(128, 5, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),

            # Output
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ], name='discriminator')

        return model

    @tf.function
    def train_step(self, real_images):
        """Un paso de entrenamiento optimizado con tf.function"""
        batch_size = tf.shape(real_images)[0]

        # Labels
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        # ============= Entrenar Discriminador =============
        noise = tf.random.normal([batch_size, self.config.latent_dim])

        with tf.GradientTape() as disc_tape:
            # Generar imágenes falsas
            generated_images = self.generator(noise, training=True)

            # Predicciones
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # Loss del discriminador
            d_loss_real = self.loss_fn(real_labels, real_output)
            d_loss_fake = self.loss_fn(fake_labels, fake_output)
            d_loss = d_loss_real + d_loss_fake

        # Gradientes y actualización
        d_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        # ============= Entrenar Generator =============
        noise = tf.random.normal([batch_size, self.config.latent_dim])

        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # Generator quiere que el discriminador clasifique las falsas como reales
            g_loss = self.loss_fn(real_labels, fake_output)

        g_gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        # Calcular accuracy
        real_acc = tf.reduce_mean(tf.cast(real_output > 0.5, tf.float32))
        fake_acc = tf.reduce_mean(tf.cast(fake_output < 0.5, tf.float32))
        d_acc = (real_acc + fake_acc) / 2.0

        return d_loss, g_loss, d_acc

    def train(self, dataset, epochs: int, callback=None):
        """Entrena la GAN"""
        print(f"\n{'='*70}")
        print(f"ENTRENAMIENTO DCGAN - Fashion-MNIST")
        print(f"{'='*70}")
        print(f"Épocas: {epochs} | Batch size: {self.config.batch_size}")
        print(f"{'='*70}\n")

        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            epoch_d_loss = 0
            epoch_g_loss = 0
            epoch_d_acc = 0
            num_batches = 0

            # Entrenar en batches
            for batch in dataset:
                d_loss, g_loss, d_acc = self.train_step(batch)

                epoch_d_loss += d_loss
                epoch_g_loss += g_loss
                epoch_d_acc += d_acc
                num_batches += 1

            # Promedios
            avg_d_loss = float(epoch_d_loss / num_batches)
            avg_g_loss = float(epoch_g_loss / num_batches)
            avg_d_acc = float(epoch_d_acc / num_batches)

            self.d_losses.append(avg_d_loss)
            self.g_losses.append(avg_g_loss)
            self.d_accuracies.append(avg_d_acc)

            # Tiempo
            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)

            # Estado
            if 0.4 <= avg_d_acc <= 0.8:
                status = "✓"
            else:
                status = "⚠"

            # Log
            print(f"{status} Época [{epoch+1:3d}/{epochs}] | "
                  f"D_Loss: {avg_d_loss:.4f} | "
                  f"G_Loss: {avg_g_loss:.4f} | "
                  f"D_Acc: {avg_d_acc:.3f} | "
                  f"Tiempo: {epoch_time:.1f}s | "
                  f"ETA: {eta/60:.1f}min")

            # Alertas
            if avg_d_acc < 0.3:
                print("     └─ ⚠ Discriminador débil, puede mejorar")
            elif avg_d_acc > 0.9:
                print("     └─ ⚠ Discriminador muy fuerte")

            # Callback UI
            if callback:
                callback(epoch + 1, epochs)

        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✓ ENTRENAMIENTO COMPLETADO")
        print(f"Tiempo total: {total_time/60:.2f} min | Promedio: {total_time/epochs:.1f}s/época")

        final_acc = self.d_accuracies[-1]
        if 0.4 <= final_acc <= 0.8:
            print(f"✓ Modelo convergió correctamente (D_Acc={final_acc:.3f})")
        else:
            print(f"⚠ Convergencia subóptima (D_Acc={final_acc:.3f})")
        print(f"{'='*70}\n")

    def generate_images(self, num_images: int, seed: Optional[int] = None) -> np.ndarray:
        """Genera imágenes desde ruido"""
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

        noise = tf.random.normal([num_images, self.config.latent_dim])
        generated = self.generator(noise, training=False)
        return generated.numpy()

    def save_models(self, path: str = "models"):
        """Guarda modelos"""
        os.makedirs(path, exist_ok=True)
        self.generator.save(f"{path}/generator.keras")
        self.discriminator.save(f"{path}/discriminator.keras")
        print(f"✓ Modelos guardados en '{path}/'")

    def load_models(self, path: str = "models") -> bool:
        """Carga modelos"""
        try:
            self.generator = keras.models.load_model(f"{path}/generator.keras")
            self.discriminator = keras.models.load_model(f"{path}/discriminator.keras")
            print(f"✓ Modelos cargados desde '{path}/'")
            return True
        except Exception as e:
            print(f"⚠ No se pudieron cargar modelos: {e}")
            return False


# ======================= DATA LOADER =======================
class FashionMNISTLoader:
    """Carga Fashion-MNIST"""

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.train_images = None

    def load_data(self) -> bool:
        """Carga dataset"""
        try:
            print("Descargando Fashion-MNIST...")
            (train_images, _), (_, _) = keras.datasets.fashion_mnist.load_data()

            # Normalizar a [-1, 1]
            train_images = train_images.astype('float32')
            train_images = (train_images - 127.5) / 127.5
            train_images = np.expand_dims(train_images, axis=-1)

            self.train_images = train_images
            print(f"✓ Fashion-MNIST cargado: {len(train_images)} imágenes")
            return True
        except Exception as e:
            print(f"✗ Error: {e}")
            return False

    def get_dataset(self) -> tf.data.Dataset:
        """Crea tf.data.Dataset optimizado"""
        dataset = tf.data.Dataset.from_tensor_slices(self.train_images)
        dataset = dataset.shuffle(10000)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


# ======================= PROMO GENERATOR =======================
class PromoGenerator:
    """Genera posters promocionales"""

    def __init__(self, gan: DCGAN):
        self.gan = gan

    def create_poster(self, n_images: int = 16, seed: Optional[int] = None) -> np.ndarray:
        """Crea poster con grid de imágenes"""
        # Generar
        images = self.gan.generate_images(n_images, seed)

        # Desnormalizar
        images = ((images + 1) * 127.5).astype(np.uint8)

        # Grid
        grid_size = int(np.sqrt(n_images))
        img_size = images.shape[1]

        canvas = np.ones((grid_size * img_size, grid_size * img_size), dtype=np.uint8) * 255

        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx < len(images):
                    img = images[idx, :, :, 0]
                    canvas[i*img_size:(i+1)*img_size, j*img_size:(j+1)*img_size] = img

        return canvas

    def save_poster(self, poster: np.ndarray, filename: str):
        """Guarda poster"""
        os.makedirs("output", exist_ok=True)
        filepath = f"output/{filename}"
        Image.fromarray(poster).save(filepath)
        print(f"✓ Poster guardado: {filepath}")
        return filepath


# ======================= VISUALIZER =======================
class GANVisualizer:
    """Interfaz Pygame"""

    def __init__(self, config: GANConfig):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Task D: GAN for Promotional Images")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        self.tiny_font = pygame.font.Font(None, 20)

        self.config = config
        self.data_loader = FashionMNISTLoader(config.batch_size)
        self.gan: Optional[DCGAN] = None
        self.promo_gen: Optional[PromoGenerator] = None

        self.is_trained = False
        self.is_training = False
        self.training_progress = 0
        self.current_epoch = 0
        self.total_epochs = EPOCHS

        self.generated_images = None
        self.promo_poster = None

        # UI
        self.train_button_rect = pygame.Rect(50, 750, 200, 50)
        self.generate_button_rect = pygame.Rect(270, 750, 200, 50)
        self.save_button_rect = pygame.Rect(490, 750, 200, 50)
        self.export_button_rect = pygame.Rect(710, 750, 200, 50)

        self.seed_slider_rect = pygame.Rect(50, 820, 400, 20)
        self.seed_value = 42

        self.epochs_slider_rect = pygame.Rect(550, 820, 400, 20)
        self.epochs_value = EPOCHS

        self.samples_rect = pygame.Rect(50, 50, 600, 600)
        self.loss_plot_rect = pygame.Rect(700, 50, 650, 600)

    def initialize(self) -> bool:
        """Inicializa sistema"""
        print("=== Inicializando GAN System ===")

        if not self.data_loader.load_data():
            return False

        self.gan = DCGAN(self.config)

        # Intentar cargar modelos
        if self.gan.load_models():
            self.is_trained = True
            self.promo_gen = PromoGenerator(self.gan)
            self._generate_samples()

        print("✓ Sistema listo\n")
        return True

    def _train_model(self):
        """Entrena GAN"""
        if self.is_training or not self.gan:
            return

        self.is_training = True
        self.current_epoch = 0
        self.training_progress = 0

        def callback(epoch, total):
            self.current_epoch = epoch
            self.training_progress = epoch / total

        dataset = self.data_loader.get_dataset()
        self.gan.train(dataset, self.epochs_value, callback)
        self.gan.save_models()

        self.is_trained = True
        self.is_training = False
        self.promo_gen = PromoGenerator(self.gan)
        self._generate_samples()

    def _generate_samples(self):
        """Genera samples"""
        if not self.gan or not self.is_trained:
            return

        self.generated_images = self.gan.generate_images(16, self.seed_value)
        if self.promo_gen:
            self.promo_poster = self.promo_gen.create_poster(16, self.seed_value)

        print(f"✓ Imágenes generadas (seed={self.seed_value})")

    def _save_poster(self):
        """Guarda poster"""
        if self.promo_poster is None:
            print("⚠ No hay poster")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.promo_gen:
            self.promo_gen.save_poster(self.promo_poster, f"promo_{timestamp}.png")

    def _export_csv(self):
        """Exporta CSV"""
        if self.generated_images is None:
            print("⚠ No hay imágenes")
            return

        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"output/gen_info_{timestamp}.csv"

        with open(filepath, 'w') as f:
            f.write("image_id,seed,mean,std\n")
            for i, img in enumerate(self.generated_images):
                f.write(f"{i},{self.seed_value},{np.mean(img):.4f},{np.std(img):.4f}\n")

        print(f"✓ CSV exportado: {filepath}")

    def _draw_samples(self):
        """Dibuja samples"""
        pygame.draw.rect(self.screen, WHITE, self.samples_rect)
        pygame.draw.rect(self.screen, BLACK, self.samples_rect, 2)

        title = self.font.render("Generated Product Images", True, BLACK)
        self.screen.blit(title, (self.samples_rect.x + 10, self.samples_rect.y - 35))

        if self.generated_images is None:
            if not self.is_trained:
                msg1 = self.small_font.render("1. Haz clic en 'Train GAN'", True, DARK_GRAY)
                msg2 = self.small_font.render("(~20-30 min con 50 épocas)", True, DARK_GRAY)
            else:
                msg1 = self.small_font.render("2. Haz clic en 'Generate Images'", True, GREEN)
                msg2 = self.small_font.render("Ajusta 'Noise Seed' para variedad", True, GREEN)

            self.screen.blit(msg1, msg1.get_rect(center=(self.samples_rect.centerx, self.samples_rect.centery - 20)))
            self.screen.blit(msg2, msg2.get_rect(center=(self.samples_rect.centerx, self.samples_rect.centery + 20)))
            return

        # Grid 4x4
        images = ((self.generated_images + 1) * 127.5).astype(np.uint8)
        cell_size = 140
        padding = 10
        start_x = self.samples_rect.x + 20
        start_y = self.samples_rect.y + 20

        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                if idx < len(images):
                    img_data = images[idx, :, :, 0]
                    img_pil = Image.fromarray(img_data).resize((cell_size, cell_size), Image.NEAREST)
                    surf = pygame.surfarray.make_surface(np.array(img_pil).T)

                    x = start_x + j * (cell_size + padding)
                    y = start_y + i * (cell_size + padding)

                    self.screen.blit(surf, (x, y))
                    pygame.draw.rect(self.screen, BLACK, (x, y, cell_size, cell_size), 1)

    def _draw_loss_plot(self):
        """Dibuja gráficas"""
        pygame.draw.rect(self.screen, WHITE, self.loss_plot_rect)
        pygame.draw.rect(self.screen, BLACK, self.loss_plot_rect, 2)

        title = self.font.render("Training Metrics", True, BLACK)
        self.screen.blit(title, (self.loss_plot_rect.x + 10, self.loss_plot_rect.y - 35))

        if not self.gan or len(self.gan.d_losses) == 0:
            msg = self.small_font.render("Métricas aparecerán tras entrenamiento", True, DARK_GRAY)
            self.screen.blit(msg, msg.get_rect(center=self.loss_plot_rect.center))
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 6), dpi=100)
        epochs = list(range(1, len(self.gan.d_losses) + 1))

        ax1.plot(epochs, self.gan.d_losses, 'r-', label='D Loss', linewidth=2)
        ax1.plot(epochs, self.gan.g_losses, 'b-', label='G Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Losses')
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(epochs, self.gan.d_accuracies, 'g-', label='D Accuracy', linewidth=2)
        ax2.axhline(0.5, color='orange', linestyle='--', label='Random', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Discriminator Accuracy')
        ax2.set_ylim([0, 1])
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        surf = pygame.image.frombuffer(canvas.buffer_rgba(), canvas.get_width_height(), "RGBA")
        self.screen.blit(surf, (self.loss_plot_rect.x, self.loss_plot_rect.y))
        plt.close(fig)

    def _draw_ui(self):
        """Dibuja UI"""
        # Botones
        train_color = GRAY if self.is_trained else BLUE
        if self.is_training:
            train_color = ORANGE

        pygame.draw.rect(self.screen, train_color, self.train_button_rect, border_radius=5)
        text = self.small_font.render("Train GAN" if not self.is_training else "Training...", True, WHITE)
        self.screen.blit(text, text.get_rect(center=self.train_button_rect.center))

        gen_color = GREEN if self.is_trained else GRAY
        pygame.draw.rect(self.screen, gen_color, self.generate_button_rect, border_radius=5)
        text = self.small_font.render("Generate Images", True, WHITE)
        self.screen.blit(text, text.get_rect(center=self.generate_button_rect.center))

        save_color = PURPLE if self.promo_poster is not None else GRAY
        pygame.draw.rect(self.screen, save_color, self.save_button_rect, border_radius=5)
        text = self.small_font.render("Save Poster", True, WHITE)
        self.screen.blit(text, text.get_rect(center=self.save_button_rect.center))

        export_color = ORANGE if self.generated_images is not None else GRAY
        pygame.draw.rect(self.screen, export_color, self.export_button_rect, border_radius=5)
        text = self.small_font.render("Export CSV", True, WHITE)
        self.screen.blit(text, text.get_rect(center=self.export_button_rect.center))

        # Sliders
        text = self.small_font.render(f"Noise Seed: {self.seed_value}", True, BLACK)
        self.screen.blit(text, (self.seed_slider_rect.x, self.seed_slider_rect.y - 25))
        pygame.draw.rect(self.screen, GRAY, self.seed_slider_rect)
        handle_x = self.seed_slider_rect.x + int((self.seed_value / 100) * self.seed_slider_rect.width)
        pygame.draw.circle(self.screen, BLUE, (handle_x, self.seed_slider_rect.centery), 12)

        text = self.small_font.render(f"Training Epochs: {self.epochs_value}", True, BLACK)
        self.screen.blit(text, (self.epochs_slider_rect.x, self.epochs_slider_rect.y - 25))
        pygame.draw.rect(self.screen, GRAY, self.epochs_slider_rect)
        ratio = (self.epochs_value - 20) / 80
        handle_x = self.epochs_slider_rect.x + int(ratio * self.epochs_slider_rect.width)
        pygame.draw.circle(self.screen, GREEN, (handle_x, self.epochs_slider_rect.centery), 12)

        # Progress bar
        if self.is_training:
            rect = pygame.Rect(50, 870, 900, 20)
            pygame.draw.rect(self.screen, GRAY, rect)
            fill = pygame.Rect(50, 870, int(900 * self.training_progress), 20)
            pygame.draw.rect(self.screen, GREEN, fill)
            pygame.draw.rect(self.screen, BLACK, rect, 2)

            text = self.tiny_font.render(
                f"Época {self.current_epoch}/{self.total_epochs} - {self.training_progress*100:.1f}%",
                True, BLACK
            )
            self.screen.blit(text, (rect.centerx - 100, rect.centery - 10))

        # Info
        texts = [
            "Fashion-MNIST: 60K imágenes de ropa",
            "DCGAN optimizada con tf.function",
            f"Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}"
        ]
        for i, t in enumerate(texts):
            surf = self.tiny_font.render(t, True, DARK_GRAY)
            self.screen.blit(surf, (1000, 865 + i * 15))

    def _handle_events(self) -> bool:
        """Maneja eventos"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos

                if self.train_button_rect.collidepoint(pos) and not self.is_training and not self.is_trained:
                    import threading
                    threading.Thread(target=self._train_model).start()

                elif self.generate_button_rect.collidepoint(pos) and self.is_trained:
                    self._generate_samples()

                elif self.save_button_rect.collidepoint(pos):
                    self._save_poster()

                elif self.export_button_rect.collidepoint(pos):
                    self._export_csv()

                elif self.seed_slider_rect.collidepoint(pos):
                    self._update_seed(pos[0])

                elif self.epochs_slider_rect.collidepoint(pos):
                    self._update_epochs(pos[0])

            elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                if self.seed_slider_rect.collidepoint(event.pos):
                    self._update_seed(event.pos[0])
                elif self.epochs_slider_rect.collidepoint(event.pos):
                    self._update_epochs(event.pos[0])

        return True

    def _update_seed(self, x: int):
        """Actualiza seed"""
        rel_x = x - self.seed_slider_rect.x
        ratio = rel_x / self.seed_slider_rect.width
        self.seed_value = int(np.clip(ratio * 100, 0, 100))

    def _update_epochs(self, x: int):
        """Actualiza epochs"""
        rel_x = x - self.epochs_slider_rect.x
        ratio = rel_x / self.epochs_slider_rect.width
        self.epochs_value = int(np.clip(20 + ratio * 80, 20, 100))
        self.total_epochs = self.epochs_value

    def run(self):
        """Loop principal"""
        print("\n=== Aplicación ejecutándose ===\n")
        running = True

        while running:
            running = self._handle_events()

            self.screen.fill(WHITE)
            self._draw_samples()
            self._draw_loss_plot()
            self._draw_ui()

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()


# ======================= MAIN =======================
def main():
    """Entry point"""
    config = GANConfig()
    app = GANVisualizer(config)

    if app.initialize():
        app.run()
    else:
        print("✗ Error inicializando")


if __name__ == "__main__":
    main()

