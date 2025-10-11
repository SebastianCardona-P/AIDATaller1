# Task C: Dynamic Pricing with Reinforcement Learning - Guía de Uso

## 📋 ¿Qué hace este sistema?

Este sistema usa **Reinforcement Learning (RL)** con el algoritmo **PPO (Proximal Policy Optimization)** para aprender a ajustar precios dinámicamente y **maximizar los ingresos** de una tienda.

El agente RL aprende a:
- Subir precios cuando la demanda es alta
- Bajar precios para estimular ventas cuando la demanda es baja
- Encontrar el balance óptimo entre precio y volumen de ventas

## 🎯 Flujo de Uso (Paso a Paso)

### **Paso 1: Ejecutar el programa**
```bash
python task_c.py
```

### **Paso 2: Entrenar el Agente RL**
1. Al iniciar, verás una pantalla con el mensaje: **"Paso 1: Haz clic en 'Train Agent' para entrenar el modelo RL"**
2. El botón **"Train Agent"** aparecerá en **GRIS** (indicando que no está entrenado)
3. Haz clic en **"Train Agent"**
4. El entrenamiento tomará **10-20 segundos** (50,000 timesteps)
5. Verás en consola: `Entrenando agente PPO para Beauty...`
6. Cuando termine: `✓ Agente entrenado con 50000 timesteps`
7. El botón cambiará a **VERDE** indicando que está listo

### **Paso 3: Configurar Precio Inicial (Opcional)**
- Usa el **slider** "Initial Price Multiplier" para ajustar el precio base
- Puedes establecer desde **0.5x** (50% descuento) hasta **2.0x** (precio doble)
- Ejemplo: Si el precio base es $100, con 1.5x iniciará en $150

### **Paso 4: Iniciar la Simulación**
1. Haz clic en **"Start/Stop"** (botón rojo que cambiará a verde)
2. La simulación comenzará a ejecutarse automáticamente
3. Verás **dos gráficas** en tiempo real:
   - **Gráfica Superior**: Comparación de ingresos (RL vs Precio Estático)
   - **Gráfica Inferior**: Evolución del precio dinámico

### **Paso 5: Observar los Resultados**
La simulación ejecutará **100 episodios** donde:
- La línea **AZUL** muestra los ingresos con RL (precio dinámico)
- La línea **ROJA** muestra los ingresos con precio estático (baseline)
- La línea **VERDE** muestra cómo el agente ajusta el precio
- La línea **NARANJA** muestra el precio base de referencia

Al finalizar, verás estadísticas:
- **Total RL Revenue**: Ingresos totales con precio dinámico
- **Total Static Revenue**: Ingresos con precio fijo
- **Improvement**: Porcentaje de mejora (¡debería ser positivo!)

## 🔄 Funciones Adicionales

### **Cambiar de Producto**
- Haz clic en los botones inferiores: **Beauty**, **Clothing**, **Electronics**
- Cada producto tiene su propio agente RL
- Deberás entrenar cada agente por separado

### **Resetear la Simulación**
- Haz clic en **"Reset"** (botón naranja)
- Esto reinicia el episodio actual pero mantiene el agente entrenado
- Útil para probar diferentes precios iniciales

### **Pausar/Continuar**
- Haz clic en **"Start/Stop"** para pausar la simulación
- Vuelve a hacer clic para continuar

## 📊 ¿Qué resultados esperar?

### **Resultados Buenos**
- **Improvement: +5% a +20%**: El RL está funcionando bien
- El precio dinámico **oscila alrededor del precio base**
- Los ingresos RL (azul) están **consistentemente por encima** de los estáticos (rojo)

### **Ejemplo de Salida en Consola**
```
✓ Episodio completado - RL: $12,450.23, Estático: $11,200.50, Mejora: 11.16%
```

### **Interpretación de las Gráficas**

#### Gráfica 1: Revenue Comparison
- **Si la línea azul está arriba de la roja**: ✅ ¡El RL está mejorando los ingresos!
- **Si ambas están muy cerca**: El RL está aprendiendo pero con margen pequeño
- **Si la línea roja está arriba**: ⚠️ Puede necesitar más entrenamiento

#### Gráfica 2: Price Evolution  
- **Precio sube y baja dinámicamente**: ✅ El agente está respondiendo a la demanda
- **Precio muy estable**: Puede que el agente sea conservador
- **Precio muy volátil**: Puede tener penalizaciones por volatilidad

## 🧠 ¿Cómo funciona el RL?

### **Estado (Observation)**
El agente observa:
1. Precio actual vs precio base
2. Promedio de ventas recientes
3. Variabilidad de ventas
4. Cantidad promedio vendida
5. Tendencia (¿están subiendo o bajando las ventas?)

### **Acción**
El agente decide un **ajuste de precio** entre -50% y +50%

### **Recompensa (Reward)**
- **Positiva**: Cuando genera más ingresos que el precio base
- **Negativa**: Por precios extremos o mucha volatilidad

### **Algoritmo: PPO (Proximal Policy Optimization)**
- Es un algoritmo de RL de última generación
- Aprende de forma estable y eficiente
- Usado en videojuegos, robótica, y sistemas de decisión

## 🔬 Experimentos Sugeridos

1. **Comparar Productos**: Entrena los 3 productos y compara cuál tiene mayor mejora
2. **Precio Inicial Alto**: Prueba con 2.0x y ve cómo el RL ajusta hacia abajo
3. **Precio Inicial Bajo**: Prueba con 0.5x y observa si el RL lo sube
4. **Re-entrenar**: Entrena varias veces el mismo producto para ver consistencia

## 🎓 Conceptos Clave

- **Elasticidad de Precio**: La demanda cae cuando el precio sube (relación inversa)
- **Trade-off**: Precio alto = menos ventas pero más ganancia por unidad
- **Exploración vs Explotación**: El agente explora diferentes precios para encontrar el óptimo
- **Policy**: La "política" es la estrategia aprendida para ajustar precios

## ⚡ Tips

- El entrenamiento es **una sola vez** por producto
- Si cambias el precio inicial después de entrenar, el agente se adaptará
- La simulación es **estocástica** (tiene aleatoriedad), así que cada ejecución será diferente
- Para mejores resultados, el agente necesita aprender patrones de demanda reales

## 🐛 Solución de Problemas

**"⚠ Por favor entrena el agente primero"**
- Haz clic en "Train Agent" antes de iniciar la simulación

**El botón "Train Agent" está en gris**
- Esto es normal antes de entrenar, haz clic en él

**La mejora es negativa**
- Puede pasar por la aleatoriedad, resetea y prueba de nuevo
- O el agente necesita más entrenamiento (aumenta TRAINING_TIMESTEPS en el código)

**La simulación va muy rápido**
- Es normal, la simulación corre a 60 FPS para completar 100 episodios rápido

---

¡Disfruta experimentando con el precio dinámico usando RL! 🚀

