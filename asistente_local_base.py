# ==============================================================================
# PROTOTIPO LOCAL DE CHATBOT (CONTROL TOTAL)
# ==============================================================================
# Este script es una plantilla base para conectar un modelo local (vía Ollama)
# utilizando Python. No envía datos a servidores externos.
#
# Requisitos previos en tu terminal:
# 1. Instalar Ollama desde ollama.com
# 2. Descargar un modelo ejecuntando: ollama run llama3.2
# 3. Instalar la librería de python: pip install requests
# ==============================================================================

import requests
import json

class MiChatGPTLocal:
    def __init__(self, model_name="llama3.2", url="http://localhost:11434/api/chat"):
        self.model_name = model_name
        self.url = url
        self.historial = []

    def enviar_mensaje(self, mensaje_usuario):
        # Añadir mensaje del usuario al historial
        self.historial.append({"role": "user", "content": mensaje_usuario})
        
        payload = {
            "model": self.model_name,
            "messages": self.historial,
            "stream": False # Cambiar a True si deseas procesar respuesta palabra por palabra
        }
        
        try:
            # Petición al servidor local de Ollama
            response = requests.post(self.url, json=payload)
            response_json = response.json()
            
            # Extraer respuesta de la IA
            respuesta_ia = response_json["message"]["content"]
            
            # Guardar en el historial para mantener el contexto de la conversación
            self.historial.append({"role": "assistant", "content": respuesta_ia})
            return respuesta_ia
            
        except requests.exceptions.ConnectionError:
            return "Error: No se pudo conectar con Ollama. Asegúrate de que Ollama está corriendo en segundo plano."

if __name__ == "__main__":
    # Inicializar chatbot (puedes cambiar "llama3.2" por "deepseek-coder", "qwen2.5", etc.)
    bot = MiChatGPTLocal(model_name="llama3.2")
    
    print("=== Mi ChatGPT Local Iniciado (Escribe 'salir' para terminar) ===")
    
    while True:
        usuario = input("\nTú: ")
        if usuario.lower() == 'salir':
            print("Cerrando entorno de desarrollo.")
            break
            
        respuesta = bot.enviar_mensaje(usuario)
        print(f"\nIA Local:\n{respuesta}")
