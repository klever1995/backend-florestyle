import firebase_admin
from firebase_admin import credentials, firestore
import traceback

def get_exact_firebase_error(json_path="florestyle2.json"):
    print("=== Probando conexión a Firebase ===\n")
    try:
        cred = credentials.Certificate(json_path)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        # Intentar listar colecciones para forzar la verificación del JWT
        collections = list(db.collections())
        print("Conexión exitosa. Colecciones:", [c.id for c in collections])
    except Exception as e:
        print("=== ERROR EXACTO DE FIREBASE ===")
        traceback.print_exc()  # Muestra todo el traceback y el mensaje exacto
        print("\nMensaje resumido:", e)

if __name__ == "__main__":
    get_exact_firebase_error()
