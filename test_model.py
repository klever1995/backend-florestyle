from ultralytics import YOLO

# Cargar el modelo desde la carpeta backend
model = YOLO("yolo11_flowers.pt")  # Ajusta la ruta si es necesario

# Ruta de prueba de una imagen (ajusta con una imagen en backend)
image_path = "C:/Users/Klever/Downloads/Gemini_Generated_Image_4x4skh4x4skh4x4s (1).jpg"  

# Realizar la predicción
results = model(image_path)

# Mostrar resultados
for result in results:
    result.show()
