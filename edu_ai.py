import sys
import os
import json
import time
import threading
import random # Moved to top
import numpy as np
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

# Importaciones de PyQt6
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QLineEdit, QLabel, 
                             QProgressBar, QGroupBox, QGridLayout, QPushButton, QScrollArea, QLayout,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QPoint, QSize
from PyQt6.QtGui import QFont, QColor, QPalette, QPainter, QBrush

# --- Configuración y Globales ---
CONFIG_PATH = "config.json"
DEFAULT_CONFIG = {
    "hidden_neurons": 32,
    "learning_rate": 0.1,
    "input_window": 5,
    "thinking_delay": 0.05,
    "visualization_mode": "heatmap"
}

try:
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    CONFIG = DEFAULT_CONFIG

# --- Funciones Auxiliares ---

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# --- El Cerebro de la IA (Red Neuronal) ---
class SimpleAI:
    """
    Implementación simple de una red neuronal para el aprendizaje de secuencias.
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        """
        Inicializa la red neuronal.

        Args:
            input_size (int): Tamaño de la capa de entrada.
            hidden_size (int): Número de neuronas en la capa oculta.
            output_size (int): Tamaño de la capa de salida (vocabulario).
            learning_rate (float): Tasa de aprendizaje para el descenso de gradiente.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.W1 = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
        self.W2 = np.random.uniform(-0.5, 0.5, (output_size, hidden_size))
        
        self.b1 = np.zeros((hidden_size, 1))
        self.b2 = np.zeros((output_size, 1))
        
        self.last_hidden = np.zeros((hidden_size, 1))
        self.last_input = None
        self.last_output = None

    @property
    def total_parameters(self):
        """Calcula el número total de parámetros (pesos + sesgos) en la red."""
        # W1 + b1 + W2 + b2
        return (self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def forward(self, inputs):
        """
        Realiza la propagación hacia adelante (forward pass).

        Args:
            inputs (np.array): Vector de entrada.
        
        Returns:
            np.array: Probabilidades de salida (Softmax).
        """
        self.last_input = inputs
        self.z1 = np.dot(self.W1, inputs) + self.b1
        self.a1 = sigmoid(self.z1)
        self.last_hidden = self.a1
        
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        self.output = softmax(self.z2)
        self.last_output = self.output
        return self.output

    def train(self, inputs, targets):
        """
        Entrena la red usando retropropagación (backpropagation).

        Args:
            inputs (np.array): Vector de entrada.
            targets (np.array): Vector objetivo (one-hot).
        """
        outputs = self.forward(inputs)
        d_z2 = outputs - targets
        d_W2 = np.dot(d_z2, self.a1.T)
        d_b2 = d_z2
        
        d_a1 = np.dot(self.W2.T, d_z2)
        d_z1 = d_a1 * sigmoid_derivative(self.a1)
        d_W1 = np.dot(d_z1, inputs.T)
        d_b1 = d_z1
        
        self.W1 -= self.learning_rate * d_W1
        self.b1 -= self.learning_rate * d_b1
        self.W2 -= self.learning_rate * d_W2
        self.b2 -= self.learning_rate * d_b2

    def save_weights(self, filepath):
        data = {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load_weights(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            
            # Validar formas antes de cargar
            W1 = np.array(data["W1"])
            b1 = np.array(data["b1"])
            W2 = np.array(data["W2"])
            b2 = np.array(data["b2"])
            
            if W1.shape != self.W1.shape or W2.shape != self.W2.shape:
                raise ValueError(f"Architecture Mismatch: Saved {W1.shape}/{W2.shape}, Current {self.W1.shape}/{self.W2.shape}")
                
            self.W2 = W2
            self.b2 = b2

    def expand_vocab(self, new_vocab_size):
        """Expande dinámicamente la arquitectura de la red para acomodar un nuevo tamaño de vocabulario."""
        if new_vocab_size <= self.output_size:
            return # Nada que hacer

        old_vocab = self.output_size
        diff = new_vocab_size - old_vocab
        window_size = int(self.input_size / old_vocab) # Inferir tamaño de ventana
        
        # 1. Expandir W1 (Entrada -> Oculta)
        # Forma W1: (Oculta, Ventana * VocabViejo)
        # Necesitamos: (Oculta, Ventana * VocabNuevo)
        # Pero simplemente añadir columnas es INCORRECTO debido a la estructura de ventana.
        # Debemos insertar 'diff' columnas cada 'old_vocab' columnas.
        
        # Redimensionar a (Oculta, Ventana, VocabViejo)
        W1_reshaped = self.W1.reshape(self.hidden_size, window_size, old_vocab)
        
        # Nuevos pesos aleatorios para nuevas palabras
        new_cols = np.random.uniform(-0.5, 0.5, (self.hidden_size, window_size, diff))
        
        # Concatenar a lo largo del eje Vocab (eje 2)
        W1_expanded = np.concatenate((W1_reshaped, new_cols), axis=2)
        
        # Aplanar de nuevo a (Oculta, Ventana * VocabNuevo)
        self.W1 = W1_expanded.reshape(self.hidden_size, window_size * new_vocab_size)
        self.input_size = self.W1.shape[1]

        # 2. Expandir W2 (Oculta -> Salida)
        # Forma W2: (VocabViejo, Oculta). Solo añadir filas.
        new_rows = np.random.uniform(-0.5, 0.5, (diff, self.hidden_size))
        self.W2 = np.vstack((self.W2, new_rows)) # 3. Expand W2 (Hidden -> Output)

        # 3. Expandir b2 (Sesgo de Salida)
        # Forma b2: (VocabViejo, 1)
        new_bias = np.zeros((diff, 1))
        self.b2 = np.vstack((self.b2, new_bias))
        
        self.output_size = new_vocab_size

    def get_weights_dict(self):
        return {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist()
        }

    def set_weights_dict(self, data):
        # Validar formas antes de cargar
        W1 = np.array(data["W1"])
        b1 = np.array(data["b1"])
        W2 = np.array(data["W2"])
        b2 = np.array(data["b2"])
        
        if W1.shape != self.W1.shape or W2.shape != self.W2.shape:
             # Intentar proveer un error útil
             raise ValueError(f"Desajuste de Arquitectura: Guardado {W1.shape}/{W2.shape}, Actual {self.W1.shape}/{self.W2.shape}")
            
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

# --- Manejador de Datos (Data Handler) ---
# (Se mantiene igual - Core Logic)
# --- Manejador de Datos (Data Handler) ---
class DataHandler:
    """Maneja el preprocesamiento de texto y la codificación/decodificación de tokens."""
    def __init__(self, mode="char"):
        self.mode = mode
        if self.mode == "char":
            self.chars = sorted([chr(i) for i in range(32, 127)] + ['\n'])
            self.token_to_ix = { ch:i for i,ch in enumerate(self.chars) }
            self.ix_to_token = { i:ch for i,ch in enumerate(self.chars) }
            self.vocab_size = len(self.chars)
        else:
            # Modo palabra: Vocabulario dinámico básico inicial
            self.tokens = ["<UNK>", "<PAD>", "\n"] 
            self.token_to_ix = { t:i for i,t in enumerate(self.tokens) }
            self.ix_to_token = { i:t for i,t in enumerate(self.tokens) }
            self.vocab_size = len(self.tokens)

    def learn_vocab(self, text):
        if self.mode == "char": return # Vocabulario estático
        
        # División simple por espacios, manteniendo saltos de línea
        words = text.replace("\n", " \n ").split(" ")
        new_tokens = set(words)
        
        added = 0
        for w in new_tokens:
            w = w.strip()
            if not w: continue
            if w not in self.token_to_ix:
                self.token_to_ix[w] = len(self.tokens)
                self.ix_to_token[len(self.tokens)] = w
                self.tokens.append(w)
                added += 1
        
        if added > 0:
            self.vocab_size = len(self.tokens)
            print(f"Vocabulario expandido: {added} nuevas palabras. Total: {self.vocab_size}")
        return added

    def encode(self, text):
        if self.mode == "char":
            return [self.token_to_ix.get(c, 0) for c in text]
        else:
            # Codificación de palabras
            words = text.replace("\n", " \n ").split(" ")
            encoded = []
            for w in words:
                w = w.strip()
                if not w: continue
                encoded.append(self.token_to_ix.get(w, 0)) # 0 es UNK
            return encoded
        
    def vector_from_token(self, token):
        x = np.zeros((self.vocab_size, 1))
        ix = self.token_to_ix.get(token, 0)
        x[ix] = 1
        return x

    def token_from_vector(self, probs):
        ix = np.argmax(probs)
        return self.ix_to_token.get(ix, "<UNK>")

    def load_file(self, path):
        try:
            ext = os.path.splitext(path)[1].lower()
            content = ""
            if ext == '.pdf':
                reader = PdfReader(path)
                for page in reader.pages:
                    content += page.extract_text() + "\n"
            else:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            
            # Auto-aprender vocabulario si está en modo palabra
            if self.mode == "word":
                self.learn_vocab(content)
                
            return content
        except Exception as e:
            return f"Error leyendo archivo: {e}"

    def load_url(self, url):
        try:
            # Añadir cabeceras para imitar navegador y evitar bloqueos (403 Forbidden)
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status() # Lanzar error si el status es malo
            
            content_type = resp.headers.get('content-type', '').lower()
            
            if url.lower().endswith('.pdf') or 'application/pdf' in content_type:
                import io
                f = io.BytesIO(resp.content)
                reader = PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            else:
                # Asumir HTML/Texto
                soup = BeautifulSoup(resp.content, 'html.parser')
                text = ' '.join([p.get_text() for p in soup.find_all('p')])
            
            # Limpieza básica y Límite para demo
            if len(text) > 50000:
                text = text[:50000] # Limitamos a 50k caracteres para no eternizar el demo
                return f"OK (Truncado a 50k chars): {text}"
            
            if self.mode == "word":
                self.learn_vocab(text)

            return text
        except Exception as e:
            return f"Error descargando URL: {e}"
    def get_state(self):
        state = {"mode": self.mode}
        if self.mode == "char":
            state["chars"] = self.chars
        else:
            state["tokens"] = self.tokens
        return state

    def set_state(self, state, force_mode=None):
        self.mode = force_mode if force_mode else state.get("mode", "char")
        if self.mode == "char":
            self.chars = state.get("chars", sorted([chr(i) for i in range(32, 127)] + ['\n']))
            self.token_to_ix = { ch:i for i,ch in enumerate(self.chars) }
            self.ix_to_token = { i:ch for i,ch in enumerate(self.chars) }
            self.vocab_size = len(self.chars)
        else:
            self.tokens = state.get("tokens", ["<UNK>", "<PAD>", "\n"])
            self.token_to_ix = { t:i for i,t in enumerate(self.tokens) }
            self.ix_to_token = { i:t for i,t in enumerate(self.tokens) }
            self.vocab_size = len(self.tokens)

from PyQt6.QtGui import QFont, QColor, QPalette, QPainter, QBrush
from PyQt6.QtCore import Qt

# --- Widgets Personalizados ---
# --- Layout de Flujo Personalizado ---
class FlowLayout(QLayout):
    """Layout que acomoda widgets en flujo (como texto), ajustándose al ancho."""
    def __init__(self, parent=None, margin=0, spacing=-1):
        super(FlowLayout, self).__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self.itemList = []

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self.itemList.append(item)

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        if index >= 0 and index < len(self.itemList):
            return self.itemList[index]
        return None

    def takeAt(self, index):
        if index >= 0 and index < len(self.itemList):
            return self.itemList.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self.doLayout(QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super(FlowLayout, self).setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())
        size += QSize(2 * self.contentsMargins().top(), 2 * self.contentsMargins().top())
        return size

    def doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        lineHeight = 0
        spacing = self.spacing()

        for item in self.itemList:
            nextX = x + item.sizeHint().width() + spacing
            if nextX - spacing > rect.right() and lineHeight > 0:
                x = rect.x()
                y = y + lineHeight + spacing
                nextX = x + item.sizeHint().width() + spacing
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y()

class NeuronVUMeter(QWidget):
    """Widget gráfico que muestra la activación de una neurona (estilo vúmetro)."""
    def __init__(self, id_num):
        super().__init__()
        self.setFixedSize(34, 110) # Mayor altura para info
        self.id = id_num
        self.value = 0.0
        self.segments = 12
        
    def set_activation(self, value):
        self.value = value
        self.repaint() # Forzar redibujado
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Dimensiones
        w = self.width()
        h = self.height()
        
        header_h = 15 # Espacio para ID (Arriba)
        footer_h = 20 # Espacio para Valor (Abajo)
        bar_h = h - header_h - footer_h
        
        # 1. Dibujar ID (Arriba)
        painter.setPen(QColor("#666"))
        painter.setFont(QFont("Courier New", 8, QFont.Weight.Bold))
        # Rect: x, y, w, h
        painter.drawText(0, 0, w, header_h, Qt.AlignmentFlag.AlignCenter, f"{self.id:02}")
        
        # 2. Dibujar Valor (Abajo)
        # Resaltar si el valor es alto
        if self.value > 0.5:
            painter.setPen(QColor("#ffffff"))
            painter.setFont(QFont("Courier New", 9, QFont.Weight.Bold))
        else:
            painter.setPen(QColor("#666"))
            painter.setFont(QFont("Courier New", 8))
            
        val_str = f"{self.value:.1f}" if self.value < 1.0 else "1.0"
        if self.value < 0.01: val_str = "." # Ruido mínimo
        
        painter.drawText(0, h - footer_h, w, footer_h, Qt.AlignmentFlag.AlignCenter, val_str)
        
        # 3. Dibujar Segmentos (Medio)
        seg_total_h = bar_h / self.segments
        gap = 1
        seg_h = seg_total_h - gap
        
        active_segments = int(self.value * self.segments)
        
        # Posición Y de inicio para la barra (tras cabecera)
        bar_start_y = header_h  
        
        for i in range(self.segments):
            # Calcular posición (De abajo hacia arriba dentro del área de la barra)
            # Fila 0 es el segmento inferior, Fila segments-1 es el superior
            row = i 
            
            # Coord Y: Base + Altura - ((fila + 1) * altura_total_seg)
            # Queremos fila 0 al fondo del área bar_h
            y = bar_start_y + bar_h - ((row + 1) * seg_total_h)
            
            # Determinar Color
            is_on = row < active_segments
            
            if not is_on:
                color = QColor("#222") # Apagado (fondo más oscuro)
            else:
                # Gradiente: Verde -> Amarillo -> Rojo
                pct = row / self.segments
                if pct < 0.6:
                    color = QColor("#00ff00") # Green
                elif pct < 0.8:
                    color = QColor("#ffff00") # Amarillo
                else:
                    color = QColor("#ff0033") # Rojo
            
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            # Margen izq/der para centrar barra
            painter.drawRect(4, int(y), w - 8, int(seg_h))

# --- Hilos de Trabajo PyQt (Workers) ---
class TrainingWorker(QThread):
    """Hilo en segundo plano para el entrenamiento de la red."""
    progress_signal = pyqtSignal(str, str) # char, buffer_completo_actual
    update_signal = pyqtSignal()

    def __init__(self, ai, data_handler, text, window_size, start_context):
        super().__init__()
        self.ai = ai
        self.data_handler = data_handler
        self.text = text
        self.window_size = window_size
        self.context_buffer = start_context

    def run(self):
        try:
            # 1. Tokenizar el texto completo primero
            # Esto nos da la lista de "unidades" de aprendizaje (chars o palabras)
            if self.data_handler.mode == "char":
                tokens = list(self.text)
            else:
                # En modo palabras, dividir preservando espacios/newlines si es necesario
                # Simplificación: Usar la misma lógica que en DataHandler
                 tokens = self.text.replace("\n", " \n ").split(" ")
                 tokens = [t.strip() for t in tokens if t.strip()]

            # Verificación de Debug
            if hasattr(self.data_handler, 'vocab_size'):
                 input_dim = self.data_handler.vocab_size * self.window_size
                 if self.ai.input_size != input_dim:
                     msg = f"CRITICO: Desajuste de Entrada Brain({self.ai.input_size}) != Data({input_dim})"
                     print(msg)
                     self.progress_signal.emit(msg, self.context_buffer)
                     # Intentar arreglar SI es estrictamente necesario, o simplemente abortar
                     return

            for token in tokens:
                if token not in self.data_handler.token_to_ix:
                    continue

                input_vec = np.zeros((self.data_handler.vocab_size * self.window_size, 1))
                
                # Codificar contexto actual
                # Aquí hay un truco: context_buffer es un string siempre?
                # Si es modo 'word', context_buffer debe ser algo que 'encode' entienda.
                # encode espera string.
                context_indices = self.data_handler.encode(self.context_buffer)
                # Asegurar que tenemos exactamente window_size indices (pad o cut)
                if len(context_indices) < self.window_size:
                    context_indices = [0]*(self.window_size - len(context_indices)) + context_indices
                else:
                    context_indices = context_indices[-self.window_size:]
                    
                for i, ix in enumerate(context_indices):
                    # Comprobación de seguridad del índice
                    if ix >= self.data_handler.vocab_size: 
                         continue
                         
                    input_vec[i * self.data_handler.vocab_size + ix] = 1
                
                target_vec = self.data_handler.vector_from_token(token)
                self.ai.train(input_vec, target_vec)
                
                # Deslizar contexto (Slide context)
                if self.data_handler.mode == "char":
                    self.context_buffer = self.context_buffer[1:] + token
                else:
                    # En modo palabra, añadimos la palabra y un espacio
                    self.context_buffer = self.context_buffer + " " + token
                    # Mantener buffer limpio para que no crezca infinito? 
                    # Optimización: Mantener solo últimas ~100 palabras en string para eficiencia
                    words = self.context_buffer.split(" ")
                    if len(words) > self.window_size + 10:
                        self.context_buffer = " ".join(words[-(self.window_size+5):])
                
                self.progress_signal.emit(token, self.context_buffer)
                self.update_signal.emit()
                
                time.sleep(max(0.001, CONFIG["thinking_delay"]))
        except Exception as e:
            print(f"WORKER ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.progress_signal.emit(f"[ERR:{e}]", self.context_buffer)

class GenerationWorker(QThread):
    """Hilo en segundo plano para la generación de texto por la IA."""
    progress_signal = pyqtSignal(str, str) # char_generado, buffer_completo_actual
    update_signal = pyqtSignal()
    finished_signal = pyqtSignal()

    def __init__(self, ai, data_handler, length, window_size, start_context):
        super().__init__()
        self.ai = ai
        self.data_handler = data_handler
        self.length = length
        self.window_size = window_size
        self.context_buffer = start_context

    def run(self):
        for _ in range(self.length):
            # Preparar input
            input_vec = np.zeros((self.data_handler.vocab_size * self.window_size, 1))
            
            context_indices = self.data_handler.encode(self.context_buffer)
            if len(context_indices) < self.window_size:
                context_indices = [0]*(self.window_size - len(context_indices)) + context_indices
            else:
                context_indices = context_indices[-self.window_size:]
                
            for i, ix in enumerate(context_indices):
                input_vec[i * self.data_handler.vocab_size + ix] = 1
            
            # Forward pass (Solo predecir, no entrenar)
            probs = self.ai.forward(input_vec)
            
            # Elegir siguiente token
            flat_probs = probs.flatten()
            ix = np.random.choice(range(len(flat_probs)), p=flat_probs)
            token = self.data_handler.ix_to_token[ix]
            
            # Actualizar contexto
            if self.data_handler.mode == "char":
                self.context_buffer = self.context_buffer[1:] + token
                display_token = token
            else:
                self.context_buffer = self.context_buffer + " " + token
                words = self.context_buffer.split(" ")
                if len(words) > self.window_size + 10:
                    self.context_buffer = " ".join(words[-(self.window_size+5):])
                display_token = " " + token # Añadir espacio para legibilidad en pantalla
            
            self.progress_signal.emit(display_token, self.context_buffer)
            self.update_signal.emit()
            
            time.sleep(max(0.001, CONFIG["thinking_delay"]))
            
        self.finished_signal.emit()

# --- Ventana Principal PyQt ---
class EduAIWindow(QMainWindow):
    """
    Ventana principal de la aplicación.
    Muestra la visualización de neuronas, predicciones y controles.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Educational AI - Neural Network Visualizer")
        self.resize(1300, 950) # AÚN MÁS GRANDE para acomodar bien las neuronas
        
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #ffffff; }
            QLabel { color: #e0e0e0; font-size: 14px; }
            QLineEdit { 
                background-color: #2b2b2b; color: #00ff00; 
                border: 1px solid #444; padding: 5px; font-family: "Courier New";
            }
            QTextEdit { 
                background-color: #121212; color: #00ff00; 
                border: 1px solid #444; font-family: "Courier New";
            }
            QGroupBox { 
                border: 1px solid #444; margin-top: 10px; font-weight: bold; color: #888;
                background-color: #252525;
            }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; }
        """)

        # Inicializar Componentes de IA
        mode = CONFIG.get("token_mode", "char")
        self.data_handler = DataHandler(mode=mode)
        self.window_size = CONFIG["input_window"]
        self.vocab_size = self.data_handler.vocab_size
        self.ai = SimpleAI(self.vocab_size * self.window_size, 
                           CONFIG["hidden_neurons"], 
                           self.vocab_size, 
                           CONFIG["learning_rate"])
        
        # Estado Persistente
        self.current_context = " " * self.window_size if mode == "char" else "" # Memoria a corto plazo
        self.worker = None 
        self.last_save_time = 0
        
        
        # Layout Principal
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # --- SECCIÓN SUPERIOR: VISUALIZACIÓN ---
        viz_layout = QHBoxLayout()
        
        # 1. Panel Izquierdo: Neuronas Ocultas (VU Meters) en SCROLL AREA
        self.neurons_group = QGroupBox("Analizador de Espectro Neuronal")
        # Layout vertical para el grupo: Contendrá el ScrollArea
        group_layout = QVBoxLayout() 
        self.neurons_group.setLayout(group_layout)
        
        # Scroll Area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border: none; background-color: transparent;")
        
        self.scroll_content = QWidget()
        self.scroll_content.setStyleSheet("background-color: transparent;")
        # USAR FLOW LAYOUT para que se acomoden solas
        self.neurons_layout = FlowLayout(self.scroll_content, margin=5, spacing=5)
        
        self.neuron_meters = []  
        
        # Grid dinámico: Ya no calculamos columnas, el FlowLayout lo hace solo
        n_neurons = CONFIG["hidden_neurons"]
        
        for i in range(n_neurons):
            meter = NeuronVUMeter(i + 1)
            self.neuron_meters.append(meter)
            self.neurons_layout.addWidget(meter)
            
        self.scroll.setWidget(self.scroll_content)
        group_layout.addWidget(self.scroll)
        
        viz_layout.addWidget(self.neurons_group, stretch=5)
        
        self.preds_group = QGroupBox("Predicciones (Top 20)")
        self.preds_layout = QVBoxLayout()
        
        self.table_preds = QTableWidget(20, 2)
        self.table_preds.setHorizontalHeaderLabels(["Token", "%"])
        self.table_preds.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_preds.verticalHeader().setVisible(False)
        self.table_preds.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_preds.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table_preds.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table_preds.setStyleSheet("""
            QTableWidget {
                background-color: transparent;
                gridline-color: #333;
                border: none;
                color: #00ff00;
                font-family: 'Courier New';
                font-size: 13px;
            }
            QHeaderView::section {
                background-color: #222;
                color: #aaa;
                border: 1px solid #333;
                font-weight: bold;
            }
        """)
        
        # Inicializar items
        for r in range(20):
            self.table_preds.setItem(r, 0, QTableWidgetItem("-"))
            self.table_preds.setItem(r, 1, QTableWidgetItem("0%"))
            self.table_preds.item(r, 0).setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_preds.item(r, 1).setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        self.preds_layout.addWidget(self.table_preds)
        self.preds_group.setLayout(self.preds_layout)
        viz_layout.addWidget(self.preds_group, stretch=1)
        
        main_layout.addLayout(viz_layout, stretch=3)
        
        # --- SECCIÓN MEDIA: INFO BAR ---
        info_layout = QHBoxLayout()
        self.lbl_config = QLabel(f"Neuronas: {CONFIG['hidden_neurons']} | LR: {CONFIG['learning_rate']}")
        self.lbl_params = QLabel("Params: 0")
        self.lbl_params.setStyleSheet("color: #00ffff; font-weight: bold; margin-left: 10px;")
        
        self.lbl_status = QLabel("Estado: ESPERANDO")
        self.lbl_status.setStyleSheet("color: #ffa500; font-weight: bold;")

        current_mode = CONFIG.get("token_mode", "char").upper()
        self.btn_switch = QPushButton(f"Modo: {current_mode} (Cambiar)")
        self.btn_switch.clicked.connect(self.toggle_mode)
        self.btn_switch.setStyleSheet("background-color: #444; color: #fff; padding: 5px; font-weight: bold;")

        info_layout.addWidget(self.lbl_config)
        info_layout.addWidget(self.lbl_params) 
        info_layout.addWidget(self.btn_switch) # Añadir botón de cambio
        info_layout.addStretch()
        info_layout.addWidget(self.lbl_status)
        main_layout.addLayout(info_layout)

        # --- SECCIÓN INFERIOR: CONTEXTO y STATS ---
        bottom_layout = QHBoxLayout()
        
        # 1. Contexto (Izquierda)
        self.memory_group = QGroupBox("Memoria de Trabajo (Stream de Contexto)")
        self.memory_layout = QVBoxLayout()
        self.txt_memory = QTextEdit()
        self.txt_memory.setReadOnly(True)
        self.txt_memory.setStyleSheet("background-color: #222; color: #ddd; font-family: 'Courier New'; font-size: 12px;")
        self.txt_memory.setPlaceholderText("IA Context Stream...")
        self.memory_layout.addWidget(self.txt_memory)
        self.memory_group.setLayout(self.memory_layout)
        
        # 2. Stats del Cerebro (Derecha, debajo de Predicciones)
        self.stats_group = QGroupBox("Estadísticas del Cerebro")
        self.stats_layout = QVBoxLayout()
        
        self.lbl_vocab = QLabel(f"Vocabulario: {self.vocab_size} tokens")
        self.lbl_vocab.setStyleSheet("color: #00ff00; font-size: 14px; font-weight: bold;")
        
        self.lbl_brain_char = QLabel("Brain (Char): ...")
        self.lbl_brain_word = QLabel("Brain (Word): ...")
        self.lbl_last_msg = QLabel("...")
        self.lbl_last_msg.setWordWrap(True)
        self.lbl_last_msg.setStyleSheet("color: #aaa; font-style: italic; font-size: 11px;")
        
        self.stats_layout.addWidget(self.lbl_vocab)
        self.stats_layout.addWidget(self.lbl_brain_char)
        self.stats_layout.addWidget(self.lbl_brain_word)
        self.stats_layout.addStretch()
        self.stats_layout.addWidget(QLabel("Ultimo Mensaje:"))
        self.stats_layout.addWidget(self.lbl_last_msg)
        
        self.stats_group.setLayout(self.stats_layout)
        
        # Añadir al layout inferior con factores de estiramiento correctos
        bottom_layout.addWidget(self.memory_group, stretch=3) # Coincide con Neuronas
        bottom_layout.addWidget(self.stats_group, stretch=1)  # Coincide con Predicciones
        
        main_layout.addLayout(bottom_layout, stretch=1)
        
        # Área de Entrada (Footer)
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Escribe texto o /gen 50, /load, /url, /cfg...")
        self.input_field.returnPressed.connect(self.handle_input)
        input_layout.addWidget(QLabel("Entrada >"))
        input_layout.addWidget(self.input_field)
        main_layout.addLayout(input_layout)

        self.log("Bienvenido. Escribe para entrenar o usa /gen 20 para dejar que la IA responda.")
        
        # AUTOLOAD: Intentar cargar cerebro con nombre correcto
        # load_brain ahora maneja el empaquetado y redimensionamiento!
        success, msg = self.load_brain()
        if success:
             self.log(f"MEMORIA CARGADA: {msg}")
        elif msg != "No file":
             self.log(f"ALERT: {msg}")
                 
        self.update_stats()
        
        # Timer de Estadísticas en Tiempo Real (bucle 2s)
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(2000)

        # Traer al frente en MacOS
        self.show()
        self.raise_()
        self.activateWindow()

    def get_brain_filename(self):
        mode = CONFIG.get("token_mode", "char")
        return f"brain_{mode}.json"

    def sync_ai_dimensions(self):
        """Asegura que la IA tiene el tamaño correcto para el vocabulario actual."""
        if self.ai.output_size < self.data_handler.vocab_size:
            added = self.data_handler.vocab_size - self.ai.output_size
            self.log(f"Sincronizando cerebro (+{added} tokens)...")
            self.ai.expand_vocab(self.data_handler.vocab_size)
            self.lbl_params.setText(f"Params: {self.ai.total_parameters:,}")
            self.update_stats()

    def save_brain(self):
        filename = self.get_brain_filename()
        try:
            state = {
                "vocab": self.data_handler.get_state(),
                "weights": self.ai.get_weights_dict(),
                "config": CONFIG
            }
            with open(filename, 'w') as f:
                json.dump(state, f)
            return filename
        except Exception as e:
            self.log(f"Error saving brain: {e}")
            raise e

    def load_brain(self):
        filename = self.get_brain_filename()
        if not os.path.exists(filename):
            return False, "No file"

        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            # Comprobar formato
            if "vocab" in data and "weights" in data:
                # 1. Restaurar Vocabulario - FORZAR modo correcto para este archivo
                target_mode = CONFIG.get("token_mode", "char")
                self.data_handler.set_state(data["vocab"], force_mode=target_mode)
                self.vocab_size = self.data_handler.vocab_size
                
                # 2. Redimensionar Arquitectura IA para coincidir con vocabulario restaurado
                self.sync_ai_dimensions()
                
                # 3. Cargar Pesos
                self.ai.set_weights_dict(data["weights"])
                
                # 4. Actualizar UI
                self.lbl_params.setText(f"Params: {self.ai.total_parameters:,}")
                self.update_stats()
                return True, f"Loaded {filename} (Bundled)"
            else:
                # Formato Legado (Solo pesos) - Fallback
                try:
                    self.ai.set_weights_dict(data) 
                    return True, f"Loaded {filename} (Legacy)"
                except ValueError as ve:
                    return False, f"Mismatch in Legacy Brain: {ve}"
                except:
                    return False, f"Legacy file format unknown"

        except Exception as e:
            return False, f"Error reading file: {e}"

    def log(self, msg):
        self.lbl_last_msg.setText(f"> {msg}")
        print(f"LOG: {msg}") # Mantener log en consola para debug
        
    def update_stats(self):
        # Actualizar Vocabulario y Modo
        mode_str = self.data_handler.mode.upper()
        self.lbl_vocab.setText(f"Vocabulario ({mode_str}): {self.data_handler.vocab_size} tokens")
        
        # Update File Sizes
        for mode in ["char", "word"]:
            fname = f"brain_{mode}.json"
            if os.path.exists(fname):
                size_mb = os.path.getsize(fname) / (1024 * 1024)
                txt = f"Brain ({mode.title()}): {size_mb:.2f} MB"
            else:
                txt = f"Brain ({mode.title()}): NO DATA"
            
            if mode == "char": self.lbl_brain_char.setText(txt)
            else: self.lbl_brain_word.setText(txt)

        # Auto-guardado periódico si el entrenamiento está activo
        now = time.time()
        if self.worker and self.worker.isRunning() and (now - self.last_save_time > 10):
            try:
                self.save_brain()
                self.last_save_time = now
            except:
                pass

    def update_viz(self):
        activations = self.ai.last_hidden.flatten()
        for i, val in enumerate(activations):
            if i < len(self.neuron_meters):
                self.neuron_meters[i].set_activation(val)
        
        if self.ai.last_output is not None:
            top_indices = np.argsort(self.ai.last_output.flatten())[-20:][::-1]
            for i, ix in enumerate(top_indices):
                if i >= self.table_preds.rowCount(): break
                token = self.data_handler.ix_to_token.get(ix, "<?>")
                prob = self.ai.last_output[ix, 0]
                
                # Update table
                token_item = self.table_preds.item(i, 0)
                prob_item = self.table_preds.item(i, 1)
                
                # Usar repr para ver espacios/newlines
                token_item.setText(repr(token))
                prob_item.setText(f"{prob*100:.2f}%")
                
                # Colorear según probabilidad (Efecto Heatmap en tabla)
                green_val = int(255 * min(1.0, prob * 5)) # Aumentar visibilidad
                color = QColor(0, max(100, green_val), 0)
                token_item.setForeground(QBrush(color))
                prob_item.setForeground(QBrush(color))

    def handle_input(self):
        text = self.input_field.text()
        self.input_field.clear()
        if not text: return
        
        if text.startswith("/"):
            self.handle_command(text)
            return

        self.start_training(text)

    def handle_command(self, cmd):
        parts = cmd.split(" ", 1)
        command = parts[0]
        arg = parts[1] if len(parts) > 1 else ""
        
        if command == "/load":
            if os.path.exists(arg):
                self.log(f"Cargando: {arg}")
                threading.Thread(target=self.load_and_train, args=(arg, "file")).start()
            else:
                self.log(f"Error: {arg} no encontrado")
        
        elif command == "/url":
            self.log(f"Descargando: {arg}")
            threading.Thread(target=self.load_and_train, args=(arg, "url")).start() # Fixed arg typo
            
        elif command == "/gen":
            try:
                length = int(arg) if arg else 20
                self.start_generation(length)
            except ValueError:
                self.log("Uso: /gen <numero de caracteres>")
                
        elif command == "/cfg":
            self.reload_config()
            
        elif command == "/quit":
            QApplication.quit()
        else:
            self.log(f"Comando desconocido: {command}")

    def load_and_train(self, path, type):
        if type == "file":
            content = self.data_handler.load_file(path)
        else:
            content = self.data_handler.load_url(path)
            
        if content.startswith("Error"):
            self.log(content)
        else:
            # Check content is not just status message
            if content.startswith("OK (Truncado"):
                real_content = content.split(": ", 1)[1]
                msg = content.split(": ", 1)[0]
                self.log(msg)
                content = real_content

            # Validar que hay caracteres útiles
            if hasattr(self.data_handler, 'char_to_ix'):
                valid_chars = sum(1 for c in content if c in self.data_handler.char_to_ix)
                if valid_chars < 10:
                    self.log(f"ADVERTENCIA: El contenido parece vacío o ilegible ({len(content)} raw, {valid_chars} valid).")
                    return
            else:
                 # En modo Word, asumimos válido si tiene longitud mínima
                 if len(content.strip()) < 10:
                      self.log(f"ADVERTENCIA: Contenido muy corto.")
                      return

            self.log(f"Datos cargados ({len(content)} chars). Entrenando...")
            QTimer.singleShot(0, lambda: self.start_training(content))

    def toggle_mode(self):
        # 1. Guardar estado actual
        try:
            filename = self.save_brain()
            self.log(f"Estado guardado en {filename}")
        except Exception as e:
            self.log(f"Error guardando estado: {e}")

        # 2. Alternar Configuración
        current_mode = CONFIG.get("token_mode", "char")
        new_mode = "word" if current_mode == "char" else "char"
        CONFIG["token_mode"] = new_mode
        
        # Guardar en config.json para persistencia entre reinicios
        try:
            with open(CONFIG_PATH, 'w') as f:
                json.dump(CONFIG, f, indent=4)
        except:
            pass
            
        self.log(f"CAMBIANDO MODO: {current_mode.upper()} -> {new_mode.upper()}...")
        
        # 3. Reiniciar todo
        # Detener cualquier worker en ejecución
        if self.worker is not None:
             if self.worker.isRunning():
                 self.worker.terminate()
                 self.worker.wait()
             self.worker = None

        self.data_handler = DataHandler(mode=new_mode)
        self.vocab_size = self.data_handler.vocab_size
        self.current_context = " " * self.window_size if new_mode == "char" else "" # Reset context
        
        self.ai = SimpleAI(self.vocab_size * self.window_size, 
                           CONFIG["hidden_neurons"], 
                           self.vocab_size, 
                           CONFIG["learning_rate"])
                           
        # 4. Cargar nuevo cerebro si existe
        # load_brain maneja empaquetado y redimensionamiento automáticamente!
        success, msg = self.load_brain()
        if success:
             self.log(f"Memoria restaurada: {msg}")
        else:
             self.log(f"Nueva memoria iniciada para {new_mode}.")
             
        # 5. Update UI
        self.lbl_params.setText(f"Params: {self.ai.total_parameters:,}")
        self.lbl_status.setText(f"Modo: {new_mode.upper()}")
        self.txt_memory.clear()
        self.log("Sistema listo en modo " + new_mode.upper())
        self.btn_switch.setText(f"Modo: {new_mode.upper()} (Cambiar)")
        self.update_stats()

    def reload_config(self):
        global CONFIG
        try:
            with open(CONFIG_PATH, 'r') as f:
                CONFIG = json.load(f)
            self.log("Config recargada.")
            
            self.lbl_config.setText(f"Neuronas: {CONFIG['hidden_neurons']} | LR: {CONFIG['learning_rate']}")
            self.window_size = CONFIG["input_window"]
            self.current_context = " " * self.window_size # Resetear contexto al cambiar config
            self.ai = SimpleAI(self.vocab_size * self.window_size, 
                               CONFIG["hidden_neurons"], 
                               self.vocab_size, 
                               CONFIG["learning_rate"])
            
            # Actualizar etiqueta de parámetros
            self.lbl_params.setText(f"Params: {self.ai.total_parameters:,}")
            
            # Recrear UI neuronas CON SCROLL
            # 1. Limpiar el widget contenido en el scroll
            old_widget = self.scroll.takeWidget()
            if old_widget: old_widget.deleteLater()
            
            self.scroll_content = QWidget()
            self.scroll_content.setStyleSheet("background-color: transparent;")
            # Layout de Flujo
            self.neurons_layout = FlowLayout(self.scroll_content, margin=5, spacing=5)
            
            self.neuron_meters = []
            
            n_neurons = CONFIG["hidden_neurons"]
            
            for i in range(n_neurons):
                meter = NeuronVUMeter(i + 1)
                self.neuron_meters.append(meter)
                self.neurons_layout.addWidget(meter)
            
            self.scroll.setWidget(self.scroll_content)
                
        except Exception as e:
            self.log(f"Error Config: {e}")

    def closeEvent(self, event):
        # Guardar cerebro al salir
        try:
            filename = self.get_brain_filename()
            self.ai.save_weights(filename)
            self.log(f"Cerebro guardado: {filename}")
        except Exception as e:
            print(f"Error guardando cerebro: {e}")
        event.accept()

    def on_worker_finished(self):
        # Si acabamos de terminar de ENTRENAR, activamos el "Balbuceo" automático
        # para que la IA intente "responder" o practicar lo aprendido.
        if isinstance(self.worker, TrainingWorker):
            self.log("Aprendido. Intentando responder...")
            # Variedad en la longitud de respuesta: Corta (15) a Larga (60)
            # Esto da más sensación de "vida"
            response_len = random.randint(15, 60)
            QTimer.singleShot(500, lambda: self.start_generation(response_len, auto=True)) 
            
        elif isinstance(self.worker, GenerationWorker):
            self.lbl_status.setText("Estado: ESPERANDO")
            self.lbl_status.setStyleSheet("color: #ffa500; font-weight: bold;")
            self.log("--- Fin de la respuesta ---")
            # No limpiamos self.worker aquí para evitar condiciones de carrera, 
            # se limpiará al iniciar la siguiente tarea.

    def start_training(self, text):
        # Asegurar que aprendemos el vocabulario del texto antes de entrenar
        if hasattr(self.data_handler, 'learn_vocab'):
             self.data_handler.learn_vocab(text)
        
        # Sincronizar dimensiones (Crecimiento dinámico)
        try:
            self.sync_ai_dimensions()
        except Exception as e:
            self.log(f"ERROR EXPANDIENDO CEREBRO: {e}")
            import traceback
            traceback.print_exc()
            return

        if self.worker is not None:
            # Si la IA está hablando (generando), la interrumpimos para que escuche (entrene)
            if self.worker.isRunning():
                self.worker.terminate() 
                self.worker.wait()      
            self.worker.deleteLater()
            self.worker = None

        self.lbl_status.setText("Estado: ENTRENANDO (Escuchando)...")
        self.lbl_status.setStyleSheet("color: #00ff00; font-weight: bold;")
        
        # CHAT FORMAT: Nueva línea para Usuario
        self.txt_memory.moveCursor(self.txt_memory.textCursor().MoveOperation.End)
        self.txt_memory.insertPlainText(f"\n\nUSER > ")
        self.txt_memory.verticalScrollBar().setValue(self.txt_memory.verticalScrollBar().maximum())
        
        self.worker = TrainingWorker(self.ai, self.data_handler, text, self.window_size, self.current_context)
        self.worker.progress_signal.connect(self.on_worker_progress)
        self.worker.update_signal.connect(self.update_viz)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()
        
    def start_generation(self, length, auto=False):
        # Sincronizar dimensiones antes de generar (Evita crash ValueError)
        self.sync_ai_dimensions()

        if self.worker is not None:
            if self.worker.isRunning():
                return 
            self.worker.deleteLater()
            self.worker = None
            
        status_msg = "Estado: BALBUCEANDO (Auto)..." if auto else "Estado: GENERANDO..."
        color = "#ff69b4" if auto else "#00ffff" 
        
        self.lbl_status.setText(status_msg)
        self.lbl_status.setStyleSheet(f"color: {color}; font-weight: bold;")
        
        # CHAT FORMAT: Nueva línea para IA
        self.txt_memory.moveCursor(self.txt_memory.textCursor().MoveOperation.End)
        prefix = "AI (Auto) > " if auto else "AI (Gen)  > "
        self.txt_memory.insertPlainText(f"\n\n{prefix}")
        self.txt_memory.verticalScrollBar().setValue(self.txt_memory.verticalScrollBar().maximum())
        
        self.worker = GenerationWorker(self.ai, self.data_handler, length, self.window_size, self.current_context)
        self.worker.progress_signal.connect(self.on_worker_progress)
        self.worker.update_signal.connect(self.update_viz)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_progress(self, char, full_buffer):
        self.current_context = full_buffer
        # Actualizar Memoria Visual
        self.txt_memory.moveCursor(self.txt_memory.textCursor().MoveOperation.End)
        self.txt_memory.insertPlainText(char)
        self.txt_memory.verticalScrollBar().setValue(self.txt_memory.verticalScrollBar().maximum())

def main():
    app = QApplication(sys.argv)
    
    # Fuentes globales
    font = QFont("Courier New", 10)
    app.setFont(font)
    
    window = EduAIWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
