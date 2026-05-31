import tkinter as tk
from tkinter import messagebox, simpledialog
import subprocess
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

PROJECT_DIR = '.'
# Detecta el Python de la venv si existe
venv_python = os.path.join(os.path.dirname(__file__), '.venv', 'Scripts', 'python.exe')
if os.path.exists(venv_python):
    PYTHON_CMD = [venv_python, '-m', 'student']
else:
    PYTHON_CMD = [sys.executable, '-m', 'student']
def run_command(args, info_msg):
    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = os.path.abspath('src')
        result = subprocess.run(args, capture_output=True, text=True, cwd=PROJECT_DIR, env=env)
        output = result.stdout if result.stdout else result.stderr
        messagebox.showinfo(info_msg, output)
    except Exception as e:
        messagebox.showerror('Error', str(e))

def index_action():
    chunk_size = simpledialog.askinteger('Chunk Size', '¿Cuántos caracteres por chunk? (1000-2000 recomendado)', initialvalue=2000, minvalue=500, maxvalue=5000)
    if chunk_size:
        args = PYTHON_CMD + ['index', '--max_chunk_size', str(chunk_size)]
        run_command(args, 'Indexing Output')

def search_action():
    query = simpledialog.askstring('Buscar', '¿Qué quieres buscar?')
    k = simpledialog.askinteger('Resultados', '¿Cuántos resultados? (default 10)', initialvalue=10, minvalue=1, maxvalue=50)
    if query:
        args = PYTHON_CMD + ['search', query, '--k', str(k)]
        run_command(args, 'Search Output')

def answer_action():
    question = simpledialog.askstring('Pregunta', '¿Qué pregunta quieres responder?')
    k = simpledialog.askinteger('Resultados', '¿Cuántos fragmentos usar? (default 10)', initialvalue=10, minvalue=1, maxvalue=50)
    if question:
        args = PYTHON_CMD + ['answer', question, '--k', str(k)]
        run_command(args, 'Answer Output')

def evaluate_action():
    answer_path = simpledialog.askstring('Ruta respuestas', 'Ruta al archivo de respuestas:')
    dataset_path = simpledialog.askstring('Ruta dataset', 'Ruta al archivo de dataset:')
    k = simpledialog.askinteger('k', '¿Valor de k? (default 10)', initialvalue=10, minvalue=1, maxvalue=50)
    if answer_path and dataset_path:
        args = PYTHON_CMD + ['evaluate', '--student_answer_path', answer_path, '--dataset_path', dataset_path, '--k', str(k)]
        run_command(args, 'Evaluation Output')

# GUI principal
root = tk.Tk()
root.title('RAG against the machine - GUI')


label.pack(pady=10)
