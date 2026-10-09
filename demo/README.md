# Demo y modelos locales

Iniciá la demo con `./run_local.sh --demo`. Si las dependencias ya están instaladas,
podés ejecutar `.venv/bin/python demo/app.py` desde la raíz del proyecto.

En la barra lateral, abrí **Modelos locales**:

1. Elegí DeepSeek-Coder 1.3B / 6.7B o Qwen2.5-Coder 3B / 7B Instruct.
2. Si figura **por descargar**, pulsá **Descargar al disco**. Esta operación
   requiere Internet y espacio para los pesos originales; la cuantización a
   4 bits se aplica al cargar en GPU, no reduce la descarga.
3. Pulsá **Cargar modelo**. Se libera el modelo anterior antes de cargar el nuevo.
   Los chats de la sesión se conservan. Cambiar la selección por sí solo no
   cambia el modelo activo: consultá el indicador **En memoria**.
4. **Liberar memoria** descarga el modelo de RAM/VRAM, sin borrar archivos.
   El próximo mensaje vuelve a cargar el último modelo utilizado.

Las descargas se guardan en `.hf_cache/hub/` dentro del repositorio, excluida de
Git. Se reutilizan los modelos ya guardados, sin copiarlos. Las cargas son
estrictamente locales y no consultan Internet. Las descargas interrumpidas pueden
reintentarse: Hugging Face reutiliza los archivos completos que ya están en caché.
No se descarga ningún modelo automáticamente al iniciar la demo.

`HF_HOME` permite conservar una ubicación de caché personalizada; `HF_HUB_CACHE`
puede especificar directamente su carpeta `hub`. `MODEL_ID` selecciona el modelo
inicial. Los modelos compatibles presentes en la caché también aparecen en la
lista. Se admiten pesos Transformers en formato Safetensors; no GGUF ni modelos
que requieran ejecutar código remoto. La restricción existente sobre DeepSeek
V2/V3/R1 se mantiene. El selector de Qwen corresponde únicamente a esta demo y
no cambia los motores de entrenamiento o la API del repositorio.

En CUDA se usan 4 bits por defecto. `QUANTIZATION=8bit` o `QUANTIZATION=none`
permiten cambiarlo; requieren más memoria. En CPU se usa float32 y en MPS,
float16. El modelo activo es compartido por las pestañas de esta instancia.
Las descargas, cargas, liberación de memoria y respuestas se procesan de forma
serial para que un cambio no interrumpa una generación en curso.

Pruebas del gestor, sin red ni GPU:

```bash
.venv/bin/python -m pytest tests/test_model_manager.py -q
```

## Chats guardados

Las conversaciones nuevas se guardan automáticamente en `.local_chats/`, un
archivo JSON UTF-8 por chat, excluido de Git. Se recuperan al abrir la demo.
Durante una respuesta se guardan avances aproximadamente cada segundo y una
última copia al terminar. Las escrituras reemplazan el archivo de forma atómica.

Para eliminar una conversación, abrila y pulsá **Borrar chat abierto**. Se borra
su archivo del disco. **Nuevo chat** conserva las conversaciones anteriores.
El panel muestra el tamaño de los archivos, el tamaño del chat abierto y los
bloques de disco asignados a los archivos de conversaciones (sin incluir modelos
ni metadatos de la carpeta). El almacenamiento se comparte entre las pestañas de
esta instalación local. Los chats que estaban únicamente en memoria antes de
instalar esta función no se pueden recuperar después de reiniciar.

## Detener y agregar contexto

El botón **+** abre un menú del mismo ancho que el cuadro de escritura.
**Archivos de texto, código o PDF** abre el selector de archivos; **Pegar texto o
instrucciones** despliega el editor. Los archivos elegidos se pueden quitar desde
la lista de adjuntos. **Listo** cierra el menú sin descartarlos.

Se admiten hasta **10 archivos**, **20 MiB por archivo** y **50 MiB en total**.
El texto adicional escrito tiene un límite de 6.000 caracteres. Los PDF requieren
`pypdf` (incluido en `demo/requirement.txt`); se extrae texto de hasta 300 páginas
con un máximo de un millón de caracteres extraídos. No incluye OCR para PDF
escaneados ni lectura de imágenes. Los archivos de código/texto deben ser UTF-8.

Los documentos largos se dividen en fragmentos y se seleccionan los que contienen
palabras de la consulta, con un presupuesto de unos 10.000 caracteres compartido
entre los adjuntos y el texto adicional. Esto no equivale a analizar el documento
completo: los extractos quedan señalados como **Fragmentos seleccionados** en el
mensaje. El contenido enviado se guarda con el chat y los adjuntos se limpian al
enviarlo. El límite de tokens del modelo sigue aplicándose a la conversación.

**■ Detener** reemplaza a la flecha durante una respuesta. Solicita detener la
generación en el siguiente paso de tokens y conserva la respuesta parcial. Si
el modelo se está cargando, la solicitud se atiende cuando termina esa carga.
No libera los pesos del modelo; para eso usá **Liberar memoria**.
