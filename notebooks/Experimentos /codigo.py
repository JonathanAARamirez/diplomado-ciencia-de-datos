import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. Cargar la imagen y forzarla siempre al formato RGB para evitar errores de canales
image_path = '/home/jonatahan/diplomado/asset/coca_roja.jpg.jpg' 
image = Image.open(image_path).convert('RGB')

# Opcional: Redimensionar para que el procesamiento y la gráfica sean fluidos en clase
# image = image.resize((150, 150)) 

# Convertir la imagen en un arreglo de NumPy con valores de 0 a 255
img_array = np.array(image)

# 2. Transformar la estructura 3D a una matriz 2D (columnas de RGB)
h, w, c = img_array.shape
X = img_array.reshape(-1, 3)

print(f"Dimensiones originales de la cuadrícula: {h} filas x {w} columnas.")
print(f"Matriz transformada para K-Means (X): {X.shape}")
print(f"Primeros 3 píxeles (RGB):\n{X[:3]}")

# 3. Mostrar la imagen original cargada
plt.figure(figsize=(5, 5))
plt.imshow(img_array)
plt.axis('off')
plt.title("Imagen Original para Extraer Colores")
plt.show()

# 4. Graficar la nube de puntos en el espacio tridimensional (RGB)
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(projection='3d')

# Extraer los canales individuales de la matriz X
r = X[:, 0]
g = X[:, 1]
b = X[:, 2]

# Colorear cada punto exactamente con su color RGB real (normalizado de 0 a 1)
pixel_colors = X / 255.0

# Dibujar la dispersión en 3D
ax.scatter(r, g, b, c=pixel_colors, s=20, alpha=0.6)

ax.set_xlabel('Canal Rojo (R)', color='red', fontweight='bold')
ax.set_ylabel('Canal Verde (G)', color='green', fontweight='bold')
ax.set_zlabel('Canal Blue (B)', color='blue', fontweight='bold')
ax.set_title('Nube de Píxeles de la Imagen en el Espacio Tridimensional (RGB)')

# Ajustar el ángulo de visión inicial para apreciar la nube en 3D
ax.view_init(elev=30, azim=45)

plt.show()












import numpy as np

# Definir el número de colores dominantes que queremos encontrar (ej. K = 4)
K = 4
epochs = 10

# 1. Inicializar centroides seleccionando K píxeles aleatorios de la matriz X
np.random.seed(42)
random_indices = np.random.choice(X.shape[0], K, replace=False)
centroids = X[random_indices].astype(float)

print(f"=== INICIANDO K-MEANS PARA CUANTIZACIÓN DE COLOR (K={K}) ===")
print(f"Centroides iniciales (Colores RGB):\n{centroids.astype(int)}\n")

for epoch in range(epochs):
    # 2. CÁLCULO DE LA MATRIZ DE DISTANCIAS EUCLIDIANAS
    # Distancia de cada píxel (N píxeles) a cada centroide (K centroides)
    # Forma resultante: (N_píxeles, K)
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    
    # 3. MATRIZ DE PERTENENCIA / ASIGNACIÓN
    # Para cada píxel, determinamos el índice del centroide más cercano (menor distancia)
    # Esto genera un vector de etiquetas de tamaño (N_píxeles,)
    labels = np.argmin(distances, axis=1)
    
    # Opcional: Mostrar un resumen matemático de la pertenencia en la primera época
    if epoch == 0:
        print(f"Dimensiones de la matriz de distancias: {distances.shape}")
        print(f"Primeros 5 registros de la Matriz de Pertenencia (Asignación a cluster): {labels[:5]}")
        print("-" * 60)
        
    # 4. ACTUALIZACIÓN DE CENTROIDES (Mover al centro de masa del cluster)
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(K)])
    
    # Condición de convergencia (si los centroides ya no se mueven)
    if np.allclose(centroids, new_centroids):
        print(f"¡K-Means convergió matemáticamente en la época {epoch + 1}!")
        break
        
    centroids = new_centroids
    print(f"Época {epoch + 1}: Centroides actualizados:\n{centroids.astype(int)}\n")

print("¡Proceso de agrupamiento por K-Means completado con éxito!")
