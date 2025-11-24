from flask import Flask, request, jsonify
from ultralytics import YOLO
import os, json
from openai import AzureOpenAI  # Cambiado
from flask_cors import CORS
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

load_dotenv()

# Configurar Firebase
firebase_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if not firebase_json:
    raise ValueError("La variable de entorno GOOGLE_APPLICATION_CREDENTIALS_JSON no está definida")

# Convertir \\n a saltos de línea reales
firebase_dict = json.loads(firebase_json)
firebase_dict["private_key"] = firebase_dict["private_key"].replace("\\n", "\n")

cred = credentials.Certificate(firebase_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()


# NUEVA CONFIGURACIÓN AZURE
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION")
)

app = Flask(__name__)
CORS(app)

MODEL_PATH = "yolo11_flowers.pt"
model = YOLO(MODEL_PATH)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# FUNCIÓN MODIFICADA PARA AZURE
def get_recommendation(flower_classes):
    try:
        # Crear clave única para esta combinación de flores
        flower_key = "_".join(sorted([flower[0] for flower in flower_classes]))
        
        # Buscar en Firebase primero
        doc_ref = db.collection('flower_recommendations').document(flower_key)
        doc = doc_ref.get()
        
        if doc.exists:
            print("✅ Recomendación desde Firebase (cache)")
            return doc.to_dict()['recommendation']
        
        # Si no existe en cache, llamar a IA
        flower_description = ", ".join([f"{flower[0]} (confianza: {flower[1]:.2f})" for flower in flower_classes])
        prompt = f"Recomendación bien concisa de cuidado para las flores detectadas: {flower_description}"
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Eres un asistente experto en flores. Responde proporcionando los cuidados necesarios para cada flor mencionada."},
                {"role": "user", "content": f"Dime cómo cuidar la flor {prompt}."} 
            ],
            max_tokens=300,
            temperature=0.2
        )

        recommendation = response.choices[0].message.content.strip()
        
        # Guardar en Firebase para próximas veces
        doc_ref.set({
            'recommendation': recommendation,
            'flower_classes': [flower[0] for flower in flower_classes],
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        print("✅ Recomendación guardada en Firebase")
        
        return recommendation if recommendation else "No se pudo obtener una recomendación."
    
    except Exception as e:
        print(f"Error al obtener la recomendación: {e}")
        return "No se pudo obtener una recomendación."

# FUNCIÓN MODIFICADA PARA AZURE
def get_flower_details(flower_classes):
    try:
        # Crear clave única para esta combinación de flores
        flower_key = "_".join(sorted([flower[0] for flower in flower_classes]))
        
        # Buscar en Firebase primero
        doc_ref = db.collection('flower_details').document(flower_key)
        doc = doc_ref.get()
        
        if doc.exists:
            print("✅ Detalles desde Firebase (cache)")
            return doc.to_dict()['details']
        
        # Si no existe en cache, llamar a IA
        flower_description = ", ".join([f"{flower[0]} (confianza: {flower[1]:.2f})" for flower in flower_classes])
        prompt = f"Proporciona detalles sobre las siguientes flores: {flower_description}. Incluye lugar de origen, nombre científico, y características principales."

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Eres un asistente experto en botánica. Responde proporcionando detalles como lugar de origen, nombre científico y características principales de las flores mencionadas."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.2
        )

        flower_details = response.choices[0].message.content.strip()
        
        # Guardar en Firebase para próximas veces
        doc_ref.set({
            'details': flower_details,
            'flower_classes': [flower[0] for flower in flower_classes],
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        print("✅ Detalles guardados en Firebase")
        
        return flower_details if flower_details else "No se pudieron obtener detalles sobre las flores."
    
    except Exception as e:
        print(f"Error al obtener los detalles de la flor: {e}")
        return "No se pudieron obtener detalles sobre las flores."

# EL RESTO DEL CÓDIGO SE MANTIENE IGUAL
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No se encontró ninguna imagen"}), 400

    image = request.files["image"]

    if not allowed_file(image.filename):
        return jsonify({"error": "Archivo no permitido, se esperan imágenes JPG, JPEG o PNG"}), 400

    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    threshold = float(request.args.get("threshold", 0.4))

    results = model(image_path)

    predictions = []
    flower_classes = []
    for result in results:
        for box in result.boxes:
            if box.conf >= threshold:
                flower_class = model.names[int(box.cls)]
                predictions.append({
                    "class": flower_class,
                    "confidence": float(box.conf)
                })
                flower_classes.append((flower_class, float(box.conf)))

    if not predictions:
        return jsonify({"message": "No se detectaron flores en la imagen. Intenta con otra imagen."}), 200

    recommendation = get_recommendation(flower_classes)
    flower_details = get_flower_details(flower_classes)

    return jsonify({
        "predictions": predictions,
        "recommendation": recommendation,
        "flower_details": flower_details
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)