# AIDATaller1 - Urban Style Retail Intelligence System

## 📋 Descripción del Proyecto

**Urban Style Retail Intelligence System** es un proyecto integral de Inteligencia Artificial aplicada al retail que combina múltiples técnicas de IA para optimizar operaciones en una tienda minorista. El proyecto implementa cuatro módulos independientes pero complementarios que abordan diferentes aspectos del negocio retail: pathfinding de clientes, forecasting de demanda, pricing dinámico con RL, y generación de imágenes promocionales con GANs.

### 🎓 Información Académica
- **Materia:** AIDA_M (Artificial Intelligence Driven by Data)
- **Institución:** Escuela Colombiana de Ingeniería Julio Garavito
- **Equipo de Desarrollo:**
  - Zayra Gutierrez
  - Andres Serrato
  - Sebastian Cardona

---

## 📊 Dataset: Retail Sales Dataset

### Descripción de los Datos
El proyecto utiliza el archivo `retail_sales_dataset.csv` que contiene transacciones de ventas minoristas con las siguientes características:

### Estructura del Dataset
| Campo | Tipo | Descripción |
|-------|------|-------------|
| Transaction ID | int | Identificador único de la transacción |
| Date | date | Fecha de la transacción (formato YYYY-MM-DD) |
| Customer ID | string | Identificador único del cliente |
| Gender | string | Género del cliente (Male/Female) |
| Age | int | Edad del cliente |
| Product Category | string | Categoría del producto |
| Quantity | int | Cantidad de productos comprados |
| Price per Unit | float | Precio unitario del producto |
| Total Amount | float | Monto total de la transacción |

### Categorías de Productos
- **Beauty**: Productos de belleza y cuidado personal
- **Clothing**: Ropa y accesorios de vestir
- **Electronics**: Productos electrónicos

### Estadísticas del Dataset
- **Total de transacciones:** 1000 registros
- **Rango temporal:** Datos de 2023
- **Clientes únicos:** Múltiples clientes identificados por CUST###

---

## 🚀 Instalación y Configuración

### Prerrequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git

### Paso 1: Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/AIDATaller1.git
cd AIDATaller1
```

### Paso 2: Crear Entorno Virtual (Recomendado)
```bash
# En Windows
python -m venv .venv
.venv\Scripts\activate

# En Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### Paso 3: Instalar Dependencias
```bash
pip install -r requirements.txt
```

### Dependencias Principales
- **pygame 2.5.2**: Motor de visualización y UI interactiva
- **tensorflow 2.15.0**: Framework de Deep Learning para LSTM y GAN
- **stable-baselines3 2.2.1**: Implementación de algoritmos de RL
- **gymnasium 0.29.1**: Biblioteca para crear entornos de RL
- **networkx 3.2.1**: Manejo de grafos para pathfinding
- **pandas 2.1.3**: Procesamiento y análisis de datos
- **matplotlib 3.8.2**: Visualización de gráficas
- **scikit-learn 1.3.2**: Preprocesamiento y métricas
- **numpy 1.26.2**: Operaciones numéricas
- **pygame-gui 0.6.9**: Componentes de interfaz gráfica
- **Pillow 10.1.0**: Procesamiento de imágenes

---

## 📁 Estructura del Proyecto

```
AIDATaller1/
├── task_a.py                    # Task A: Customer Pathfinding
├── task_b.py                    # Task B: Demand Forecasting
├── task_c.py                    # Task C: Dynamic Pricing
├── task_d.py                    # Task D: GAN Image Generation
├── retail_sales_dataset.csv     # Dataset principal
├── requirements.txt             # Dependencias del proyecto
├── README.md                    # Documentación
├── models/                      # Modelos entrenados
│   ├── generator.keras
│   └── discriminator.keras
├── output/                      # Imágenes generadas
│   └── promo_*.png
└── img/                         # Recursos visuales
```

---

## 🎯 Task A: Customer Pathfinding & Store Layout

### Descripción
Simulación interactiva de movimiento de clientes dentro de una tienda minorista usando algoritmos de búsqueda para identificar rutas óptimas y zonas de alto tráfico.

### 🔧 Implementación Técnica

#### Arquitectura del Código
```python
# Clases principales:
- StoreGrid: Representa la tienda como una cuadrícula 8x6
- Customer: Representa un cliente con posición y objetivo
- PathfindingAlgorithm: Clase abstracta para algoritmos
- BFSPathfinding: Implementación de Breadth-First Search
- AStarPathfinding: Implementación de A* con heurística Manhattan
- TrafficMonitor: Monitorea y visualiza zonas de alto tráfico
- StoreSimulation: Controlador principal
- GameVisualizer: Interfaz Pygame
```

#### Características Implementadas
1. **Grid de Tienda (8x6 celdas)**
   - Entrada en posición (0, 3)
   - 3 secciones de productos:
     - Beauty (fila superior)
     - Clothing (fila media)
     - Electronics (fila inferior)

2. **Algoritmos de Pathfinding**
   - **BFS (Breadth-First Search)**: Explora nivel por nivel, garantiza ruta más corta
   - **A* (A-Star)**: Usa heurística Manhattan para búsqueda dirigida y eficiente
   - Ambos evitan obstáculos y encuentran rutas óptimas

3. **Sistema de Clientes**
   - Múltiples clientes simultáneos en la tienda
   - Movimiento realista con velocidad ajustable
   - Los clientes permanecen tiempo aleatorio en cada sección (15-45 segundos)
   - Comportamiento natural de compra

4. **Monitoreo de Tráfico**
   - Mapa de calor visual de zonas de alto tráfico
   - Colores indican intensidad de visitas (verde→amarillo→rojo)
   - Útil para optimización de layout

#### Controles Interactivos
- **Botón "Switch Algorithm"**: Alterna entre BFS y A*
- **Botón "Change Customer Count"**: Cambia cantidad de clientes
- **Visualización en tiempo real**: Grid, clientes (puntos), secciones coloreadas

### 🎮 Cómo Ejecutar
```bash
python task_a.py
```

**Flujo de uso:**
1. La aplicación inicia con 5 clientes usando algoritmo BFS
2. Observa cómo los clientes se mueven hacia las secciones
3. Haz clic en "Switch Algorithm" para cambiar a A* y comparar
4. Haz clic en "Change Customer Count" para modificar la cantidad
5. Observa el mapa de calor para identificar zonas populares

### 📊 Principios de Diseño
- **SOLID**: Cada clase tiene una responsabilidad única
- **Strategy Pattern**: Algoritmos intercambiables
- **Clean Code**: Nombres descriptivos, funciones pequeñas
- **Modularidad**: Componentes independientes y reutilizables

---

## 📈 Task B: Demand Forecasting (Time-Series LSTM)

### Descripción
Sistema de predicción de demanda usando redes neuronales LSTM para forecasting de ventas semanales por categoría de producto.

### 🔧 Implementación Técnica

#### Arquitectura del Código
```python
# Clases principales:
- DataLoader: Carga y valida datos del CSV
- TimeSeriesPreprocessor: Convierte datos a series temporales semanales
- LSTMModel: Red neuronal LSTM para forecasting
- ModelTrainer: Entrenamiento y evaluación
- PredictionEngine: Genera predicciones futuras
- ForecastVisualizer: Interfaz Pygame con gráficas
```

#### Características Implementadas
1. **Preprocesamiento de Datos**
   - Conversión a serie temporal semanal
   - Normalización con MinMaxScaler (0-1)
   - Creación de secuencias con ventana temporal (lookback=4 semanas)
   - Split 80/20 para train/test

2. **Arquitectura LSTM**
   ```
   Input (4 semanas) 
   → LSTM(64 units, return_sequences=True)
   → Dropout(0.3)
   → LSTM(32 units)
   → Dropout(0.3)
   → Dense(16, relu)
   → Dense(1) [Predicción]
   ```
   - 2 capas LSTM apiladas para capturar patrones temporales complejos
   - Dropout para regularización y prevenir overfitting
   - Optimizador: Adam
   - Loss: Mean Squared Error (MSE)

3. **Métricas de Evaluación**
   - **MAE (Mean Absolute Error)**: Error promedio absoluto
   - **RMSE (Root Mean Squared Error)**: Error cuadrático medio
   - Valores logrados: MAE ~689-720, RMSE ~819-842
   - Interpretación: Errores aceptables considerando escala de ventas

4. **Predicción**
   - Forecasting de próximas 4-8 semanas
   - Visualización con intervalos de confianza
   - Exportación a CSV

#### Controles Interactivos
- **Slider "Category"**: Selecciona categoría (Beauty, Clothing, Electronics)
- **Slider "Training Epochs"**: Ajusta épocas de entrenamiento (50-1000)
- **Botón "Train Model"**: Inicia entrenamiento
- **Botón "Export Predictions"**: Guarda predicciones en CSV
- **Gráfica interactiva**: Muestra datos históricos y predicciones

### 🎮 Cómo Ejecutar
```bash
python task_b.py
```

**Flujo de uso:**
1. Selecciona una categoría con el slider
2. Ajusta las épocas de entrenamiento (más épocas = mejor precisión pero más tiempo)
3. Haz clic en "Train Model" y espera (puede tomar varios minutos)
4. Observa las métricas MAE y RMSE en consola
5. Analiza la gráfica de predicción vs datos reales
6. Exporta predicciones si es necesario

### 📊 Interpretación de Resultados

**¿Son buenos los resultados?**
- **MAE ~700-720**: El modelo se equivoca en promedio por ~700 unidades
- **RMSE ~820-840**: Penaliza errores grandes
- Dado que las ventas semanales promedian 3000-5000 unidades, un error de 700 representa ~15-20% de error, lo cual es **razonable para forecasting**.

**Mejoras posibles:**
- Más datos históricos (actualmente ~53 semanas)
- Features adicionales (promociones, estacionalidad, días festivos)
- Arquitecturas más complejas (GRU, Transformer)
- Ensemble de modelos

---

## 💰 Task C: Dynamic Pricing with Reinforcement Learning

### Descripción
Sistema de pricing dinámico que usa Reinforcement Learning (PPO) para ajustar precios automáticamente y maximizar ingresos.

### 🔧 Implementación Técnica

#### Arquitectura del Código
```python
# Clases principales:
- DataProcessor: Procesa datos de productos y ventas
- PricingEnvironment(gym.Env): Ambiente Gymnasium personalizado
- RLAgent: Agente PPO de Stable-Baselines3
- SimulationEngine: Motor de simulación de ventas
- RevenueTracker: Monitorea métricas de revenue
- DynamicPricingVisualizer: Interfaz Pygame
```

#### Características Implementadas
1. **Ambiente de Reinforcement Learning**
   - **State Space (Box):**
     - Precio actual normalizado
     - Precio promedio histórico
     - Ventas recientes (últimos 5 pasos)
     - Precio máximo/mínimo del producto
     
   - **Action Space (Discrete 5):**
     - 0: Reducir precio 10%
     - 1: Reducir precio 5%
     - 2: Mantener precio
     - 3: Aumentar precio 5%
     - 4: Aumentar precio 10%
   
   - **Reward Function:**
     ```
     reward = revenue - penalty_por_precio_extremo
     ```
     - Maximiza ingresos
     - Penaliza precios muy altos o muy bajos
     - Considera elasticidad de demanda

2. **Algoritmo PPO (Proximal Policy Optimization)**
   - **Hyperparámetros:**
     - learning_rate: 0.0003
     - n_steps: 2048
     - batch_size: 64
     - n_epochs: 10
     - gamma: 0.99
   - **Training:** 50,000 timesteps por producto
   - **Policy Network:** MLP [64, 64]

3. **Simulación de Demanda**
   - Elasticidad precio-demanda realista
   - Demanda base extraída del dataset
   - Variación aleatoria de mercado
   - Comparación con precio estático

4. **Métricas de Evaluación**
   - Revenue acumulado (dinámico vs estático)
   - Porcentaje de mejora sobre precio fijo
   - Historial de precios y ventas
   - Convergencia del aprendizaje

#### Controles Interactivos
- **Dropdown "Select Product"**: Elige categoría (Beauty, Clothing, Electronics)
- **Slider "Initial Price"**: Establece precio base ($50-$500)
- **Botón "Train Agent"**: Entrena agente RL (toma ~2-3 minutos)
- **Botón "Reset Simulation"**: Reinicia con nuevo precio base
- **Gráfica en tiempo real**: Muestra evolución de ingresos

### 🎮 Cómo Ejecutar
```bash
python task_c.py
```

**Flujo de uso:**
1. Selecciona un producto del dropdown
2. Ajusta el precio inicial con el slider
3. Haz clic en "Train Agent" (IMPORTANTE: debes entrenar primero)
4. Espera a que termine el entrenamiento (verás "✓ Agente entrenado")
5. Observa la gráfica de ingresos comparando RL vs Precio Estático
6. Experimenta con "Reset Simulation" para ver diferentes escenarios

### 📊 Resultados Esperados
- **Revenue Improvement:** 10-30% sobre precio estático
- **Comportamiento del Agente:**
  - Baja precios cuando demanda cae
  - Sube precios cuando demanda es alta
  - Encuentra precio óptimo dinámicamente

### ⚠️ Nota Importante
**SIEMPRE entrena el agente primero** antes de usar la simulación. Si ves "⚠ Por favor entrena el agente primero", haz clic en "Train Agent".

---

## 🎨 Task D: GAN for Product Promotional Images

### Descripción
Generación de imágenes promocionales de productos usando Generative Adversarial Networks (DCGAN) entrenada con Fashion-MNIST.

### 🔧 Implementación Técnica

#### Arquitectura del Código
```python
# Clases principales:
- FashionMNISTLoader: Carga dataset Fashion-MNIST
- Generator: Red generativa (noise → image)
- Discriminator: Red discriminativa (image → real/fake)
- GANModel: Modelo GAN completo
- GANTrainer: Entrenamiento adversarial
- ImageGenerator: Genera imágenes con seeds
- PromoCreator: Crea posters promocionales
- GANVisualizer: Interfaz Pygame
```

#### Características Implementadas
1. **Arquitectura DCGAN (Deep Convolutional GAN)**

   **Generator:**
   ```
   Input: Noise (100D) 
   → Dense(7×7×256) → Reshape(7,7,256)
   → Conv2DTranspose(128, 5×5, stride=1) → BatchNorm → LeakyReLU
   → Conv2DTranspose(64, 5×5, stride=2) → BatchNorm → LeakyReLU
   → Conv2DTranspose(1, 5×5, stride=2, tanh)
   Output: Image (28×28×1)
   ```
   - Convierte ruido aleatorio en imágenes realistas
   - BatchNormalization para estabilidad
   - Activación Tanh en output [-1, 1]

   **Discriminator:**
   ```
   Input: Image (28×28×1)
   → Conv2D(64, 5×5, stride=2) → LeakyReLU → Dropout(0.3)
   → Conv2D(128, 5×5, stride=2) → LeakyReLU → Dropout(0.3)
   → Flatten → Dense(1, sigmoid)
   Output: Probabilidad [0=fake, 1=real]
   ```
   - Clasifica imágenes como reales o generadas
   - Dropout para regularización

2. **Proceso de Entrenamiento**
   - **Dataset:** Fashion-MNIST (60,000 imágenes de ropa)
   - **Batch Size:** 128
   - **Épocas:** 20-50 (configurable)
   - **Optimizadores:** Adam con β1=0.5
   - **Learning Rates:** 
     - Generator: 0.0002
     - Discriminator: 0.0001
   
   - **Entrenamiento Adversarial:**
     ```
     for epoch in epochs:
         for batch in data:
             # 1. Entrenar Discriminator
             real_images → D → loss_real
             noise → G → fake_images → D → loss_fake
             loss_D = loss_real + loss_fake
             
             # 2. Entrenar Generator
             noise → G → fake_images → D → loss_G
     ```

3. **Métricas de Monitoreo**
   - **D_Loss:** Pérdida del discriminador (debe estar ~0.5-0.8)
   - **G_Loss:** Pérdida del generador (debe disminuir)
   - **D_Acc:** Precisión del discriminador (~0.5-0.7 es ideal)
   - **ETA:** Tiempo estimado restante
   - **Alertas:** Detecta mode collapse o desbalance

4. **Generación de Posters Promocionales**
   - Genera múltiples imágenes con diferentes seeds
   - Combina en layout de poster 2×3
   - Añade texto promocional
   - Guarda en alta resolución

#### Controles Interactivos
- **Slider "Training Epochs"**: Épocas de entrenamiento (10-50)
- **Slider "Noise Seed"**: Semilla para generación reproducible
- **Botón "Train GAN"**: Entrena modelo desde cero
- **Botón "Generate Images"**: Genera imágenes con seed actual
- **Botón "Create Promo Poster"**: Crea poster promocional completo
- **Progress Bar**: Muestra progreso de entrenamiento
- **Preview**: Vista previa de imágenes generadas

### 🎮 Cómo Ejecutar
```bash
python task_d.py
```

**Flujo de uso:**
1. Ajusta épocas de entrenamiento (20-30 recomendado)
2. Haz clic en "Train GAN" (puede tomar 1-6 horas según épocas y hardware)
3. Observa métricas en consola:
   ```
   Época [5/20] | D_Loss: 0.6234 | G_Loss: 2.1456 | D_Acc: 0.654
   ```
4. Espera a que termine (verás "✓ ENTRENAMIENTO COMPLETADO!")
5. Ajusta el seed slider para diferentes resultados
6. Haz clic en "Generate Images" para ver productos generados
7. Haz clic en "Create Promo Poster" para crear poster promocional
8. Encuentra el poster en la carpeta `output/`

### 📊 Interpretación de Resultados

**Métricas durante entrenamiento:**
- **D_Acc cerca de 0.5-0.7**: ✅ Perfecto balance, ambas redes compitiendo
- **D_Acc > 0.9**: ⚠️ Discriminador muy fuerte, generador no aprende
- **D_Acc < 0.3**: ⚠️ Generador dominando, colapso inminente
- **G_Loss disminuyendo gradualmente**: ✅ Generador mejorando
- **D_Loss estable ~0.5-0.8**: ✅ Entrenamiento saludable

**Calidad de imágenes:**
Las imágenes generadas representan prendas de ropa del Fashion-MNIST:
- Camisetas (T-shirts/tops)
- Pantalones (Trousers)
- Vestidos (Dresses)
- Zapatos (Shoes)
- Bolsos (Bags)
- Etc.

**Nota:** Las imágenes son en escala de grises 28×28 píxeles, simples pero reconocibles como prendas de vestir.

### ⏱️ Tiempo de Entrenamiento
- **10 épocas:** ~30-60 minutos
- **20 épocas:** ~1-2 horas
- **30 épocas:** ~2-3 horas
- **50 épocas:** ~4-6 horas

*Tiempos varían según hardware (GPU acelera significativamente)*

---

## 🎯 Principios de Diseño y Mejores Prácticas

### Todos los Tasks Implementan:

#### 1. **SOLID Principles**
- **S (Single Responsibility):** Cada clase tiene una única responsabilidad
- **O (Open/Closed):** Extensible sin modificar código existente
- **L (Liskov Substitution):** Clases derivadas sustituibles
- **I (Interface Segregation):** Interfaces específicas
- **D (Dependency Inversion):** Dependencias a abstracciones

#### 2. **Clean Code**
- Nombres descriptivos y significativos
- Funciones pequeñas y enfocadas
- Comentarios solo cuando necesario
- DRY (Don't Repeat Yourself)
- Formateo consistente

#### 3. **Design Patterns**
- **Strategy:** Algoritmos intercambiables (Task A)
- **Observer:** Monitoreo de eventos
- **Factory:** Creación de objetos
- **Singleton:** Instancias únicas donde apropiado

#### 4. **Modularidad**
- Cada task en archivo independiente
- Clases cohesivas y bajo acoplamiento
- Fácil de mantener y extender
- Reutilización de componentes

#### 5. **Type Hints**
```python
def process_data(data: pd.DataFrame) -> np.ndarray:
    """Procesa datos y retorna array."""
    pass
```

#### 6. **Dataclasses**
```python
@dataclass
class Config:
    learning_rate: float = 0.001
    epochs: int = 100
```

---

## 🐛 Troubleshooting

### Problemas Comunes

#### 1. Error de importación de módulos
```bash
# Solución: Reinstalar dependencias
pip install -r requirements.txt --force-reinstall
```

#### 2. TensorFlow GPU no funciona
```bash
# Verifica instalación de CUDA y cuDNN
# O usa versión CPU: tensorflow-cpu
```

#### 3. Pygame no inicia
```bash
# Windows: Instala Visual C++ Redistributables
# Linux: sudo apt-get install python3-pygame
```

#### 4. Memoria insuficiente (Task D)
- Reduce batch_size en código
- Reduce número de épocas
- Cierra otras aplicaciones

#### 5. "Please train agent first" (Task C)
- SIEMPRE haz clic en "Train Agent" antes de usar
- Espera a ver "✓ Agente entrenado"

---

## 📚 Referencias y Recursos

### Papers y Recursos Académicos
- **A* Algorithm:** Hart, P. E., Nilsson, N. J., & Raphael, B. (1968)
- **LSTM Networks:** Hochreiter & Schmidhuber (1997)
- **PPO:** Schulman et al. (2017) - Proximal Policy Optimization
- **DCGAN:** Radford et al. (2015) - Unsupervised Representation Learning
- **Fashion-MNIST:** Xiao, Han, et al. (2017)

### Librerías Utilizadas
- [Pygame Documentation](https://www.pygame.org/docs/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [NetworkX Guide](https://networkx.org/documentation/stable/)

---

## 🔮 Futuras Mejoras

### Task A
- [ ] Agregar más tipos de productos
- [ ] Implementar Dijkstra y otros algoritmos
- [ ] Análisis predictivo de tráfico
- [ ] Optimización automática de layout

### Task B
- [ ] Incorporar variables exógenas (clima, promociones)
- [ ] Modelos ensemble (LSTM + XGBoost)
- [ ] Forecasting multivariado
- [ ] Dashboard interactivo avanzado

### Task C
- [ ] Múltiples agentes compitiendo
- [ ] Consideration de competidores
- [ ] Pricing personalizado por segmento
- [ ] A/B testing automatizado

### Task D
- [ ] Conditional GAN para categorías específicas
- [ ] StyleGAN para mayor resolución
- [ ] Text-to-Image con prompts
- [ ] Generación de videos promocionales

---

## 📄 Licencia

Este proyecto es desarrollado con fines académicos para la materia AIDA_M.

---

## 👥 Contacto

**Equipo de Desarrollo:**
- Zayra Gutierrez
- Andres Serrato
- Sebastian Cardona

**Repositorio:** [AIDATaller1](https://github.com/SebastianCardona-P/AIDATaller1)

---


**Última actualización:** Octubre 2025

