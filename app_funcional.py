#!/usr/bin/env python3
"""TraduLibras - Sistema de reconhecimento LIBRAS"""

from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import tempfile
import threading
import time
from gtts import gTTS
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'tradulibras_secret_key_2024'

# Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Simulação simples do user_manager para evitar erros
class SimpleUserManager:
    def get_user(self, user_id):
        return User(user_id, f"user{user_id}", False)
    
    def authenticate(self, username, password):
        if username == "admin" and password == "admin":
            return User(1, "admin", True)
        return None
    
    def get_stats(self):
        return {"total_users": 1, "active_sessions": 1}

class User:
    def __init__(self, id, username, is_admin=False):
        self.id = id
        self.username = username
        self.is_admin = is_admin
    
    def is_authenticated(self):
        return True
    
    def is_active(self):
        return True
    
    def is_anonymous(self):
        return False
    
    def get_id(self):
        return str(self.id)
    
    def is_admin(self):
        return self.is_admin

user_manager = SimpleUserManager()

@login_manager.user_loader
def load_user(user_id):
    return user_manager.get_user(user_id)

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Carregar modelo
pasta_modelos = 'modelos/'
modelo_file = os.path.join(pasta_modelos, 'modelo_libras.pkl')
gestos_file = os.path.join(pasta_modelos, 'gestos_treinados.txt')

# Inicializar variáveis
model = None
gestos_treinados = []

try:
    if os.path.exists(modelo_file):
        print("📦 Carregando modelo...")
        with open(modelo_file, 'rb') as f: 
            model = pickle.load(f)
        print("✅ Modelo carregado com sucesso!")
        
        if os.path.exists(gestos_file):
            with open(gestos_file, 'r') as f:
                gestos_treinados = f.read().strip().split(',')
            print(f"🎯 Gestos treinados: {gestos_treinados}")
        else:
            # Tentar inferir do modelo
            if hasattr(model, 'classes_'):
                gestos_treinados = model.classes_.tolist()
            else:
                gestos_treinados = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
                                  'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                                  'ESPACO','PONTO']
                
except Exception as e:
    print(f"❌ Erro ao carregar modelo: {e}")
    model = None

# Variáveis globais
current_letter = ""
formed_text = ""
last_prediction_time = datetime.now()
hand_detected_time = None
prediction_cooldown = 2.0
min_hand_time = 1.0
auto_speak_enabled = True
confidence_threshold = 0.3
selected_camera_index = 0

def extrair_caracteristicas(hand_landmarks):
    """Extrai 63 features - compatível com o coletor"""
    if not hand_landmarks:
        return None
    
    p0 = hand_landmarks.landmark[0]  # Pulso
    dados = []
    for lm in hand_landmarks.landmark:
        dados.extend([lm.x - p0.x, lm.y - p0.y, lm.z - p0.z])
    return dados

def detectar_webcam_usb():
    """Detectar webcam automaticamente"""
    for i in range(5):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened() and cap.read()[0]:
                cap.release()
                print(f"📹 Webcam detectada: índice {i}")
                return i
        except:
            pass
    print("⚠️  Usando webcam padrão (índice 0)")
    return 0

selected_camera_index = detectar_webcam_usb()

def generate_frames():
    global current_letter, formed_text, last_prediction_time, hand_detected_time
    
    camera = cv2.VideoCapture(selected_camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not camera.isOpened():
        print("❌ Erro: Não foi possível acessar a câmera")
        return
    
    while True:
        try:
            success, frame = camera.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            current_time = datetime.now()
            
            if results.multi_hand_landmarks:
                if hand_detected_time is None:
                    hand_detected_time = current_time
                
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    points = extrair_caracteristicas(hand_landmarks)
                
                time_since_detection = (current_time - hand_detected_time).total_seconds()
                
                if (time_since_detection >= min_hand_time and 
                    points and len(points) == 63 and 
                    model is not None):
                    
                    time_since_last = (current_time - last_prediction_time).total_seconds()
                    
                    if time_since_last >= prediction_cooldown:
                        try:
                            predicted_proba = model.predict_proba([points])[0]
                            max_confidence = np.max(predicted_proba)
                            predicted_index = np.argmax(predicted_proba)
                            
                            if max_confidence >= confidence_threshold:
                                predicted_gesto = gestos_treinados[predicted_index]
                                
                                print(f"🎯 {predicted_gesto} ({max_confidence*100:.1f}%)")
                                
                                # Processar gestos especiais
                                if predicted_gesto == 'ESPACO':
                                    current_letter = '[ESPAÇO]'
                                    formed_text += ' '
                                elif predicted_gesto == 'PONTO':
                                    current_letter = '[PONTO]'
                                    formed_text += '.'
                                    # Falar automaticamente se habilitado
                                    if auto_speak_enabled and formed_text.strip():
                                        texto_limpo = formed_text.strip()
                                        threading.Thread(
                                            target=falar_texto_automatico, 
                                            args=(texto_limpo,), 
                                            daemon=True
                                        ).start()
                                else:
                                    current_letter = predicted_gesto
                                    formed_text += predicted_gesto
                                
                                last_prediction_time = current_time
                                hand_detected_time = None
                                
                            else:
                                # Confiança baixa
                                current_letter = "?"
                                
                        except Exception as e:
                            print(f"❌ Erro na predição: {e}")
            else:
                hand_detected_time = None
                current_letter = ""
            
            # Codificar frame para streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                break
                
        except Exception as e:
            print(f"❌ Erro no generate_frames: {e}")
            break
    
    camera.release()

def falar_texto_automatico(texto):
    """Falar texto usando gTTS"""
    try:
        if not texto.strip():
            return
            
        tts = gTTS(text=texto, lang='pt-br')
        temp_file = os.path.join(tempfile.gettempdir(), f'tradulibras_{int(time.time())}.mp3')
        tts.save(temp_file)
        
        # Limpar arquivo temporário após algum tempo
        def limpar_arquivo(file_path):
            time.sleep(30)
            if os.path.exists(file_path):
                os.remove(file_path)
        
        threading.Thread(target=limpar_arquivo, args=(temp_file,), daemon=True).start()
        
    except Exception as e:
        print(f"❌ Erro no TTS: {e}")

# Rotas de autenticação
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('camera_tradulibras'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        user = user_manager.authenticate(username, password)
        if user:
            login_user(user)
            flash('Login realizado com sucesso!', 'success')
            return redirect(url_for('camera_tradulibras'))
        else:
            flash('Usuário ou senha incorretos!', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Desconectado com sucesso!', 'success')
    return redirect(url_for('login'))

# Rotas principais
@app.route('/camera')
@login_required
def camera_tradulibras():
    return render_template('camera_tradulibras.html', 
                         modelo_carregado=model is not None,
                         gestos_treinados=gestos_treinados,
                         limiar_confianca=confidence_threshold)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Rotas de API
@app.route('/letra_atual')
@login_required
def get_letra_atual():
    return jsonify({
        "letra": current_letter,
        "texto": formed_text
    })

@app.route('/limpar_texto', methods=['POST'])
@login_required
def limpar_texto_completo():
    global formed_text, current_letter
    formed_text = ""
    current_letter = ""
    return jsonify({"status": "success", "texto": formed_text})

@app.route('/limpar_ultima_letra', methods=['POST'])
@login_required
def limpar_ultima_letra():
    global formed_text, current_letter
    if formed_text:
        formed_text = formed_text[:-1]
        current_letter = ""
    return jsonify({"status": "success", "texto": formed_text})

@app.route('/falar_texto', methods=['POST'])
@login_required
def falar_texto():
    if formed_text.strip():
        try:
            falar_texto_automatico(formed_text)
            return jsonify({"success": True, "message": "Texto enviado para fala"})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    return jsonify({"success": False, "error": "Texto vazio"})

@app.route('/auto_speak/toggle', methods=['POST'])
@login_required
def toggle_auto_speak():
    global auto_speak_enabled
    data = request.get_json()
    if data and 'enabled' in data:
        auto_speak_enabled = bool(data['enabled'])
    return jsonify({
        'success': True, 
        'auto_speak_enabled': auto_speak_enabled
    })

@app.route('/ajustar_limiar', methods=['POST'])
@login_required
def ajustar_limiar_confianca():
    global confidence_threshold
    try:
        data = request.get_json()
        if data and 'limiar' in data:
            novo_limiar = float(data['limiar'])
            if 0.1 <= novo_limiar <= 0.95:
                confidence_threshold = novo_limiar
                return jsonify({'success': True, 'limiar': confidence_threshold})
        return jsonify({'success': False, 'message': 'Limiar inválido'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Erro: {str(e)}'})

@app.route('/status')
@login_required
def status():
    return jsonify({
        "modelo_carregado": model is not None,
        "gestos_treinados": gestos_treinados,
        "texto_atual": formed_text,
        "letra_atual": current_letter,
        "limiar_confianca": confidence_threshold,
        "auto_speak": auto_speak_enabled,
        "cooldown": prediction_cooldown
    })

# Rota de diagnóstico
@app.route('/diagnostico')
@login_required
def diagnostico():
    return jsonify({
        'modelo_carregado': model is not None,
        'total_gestos': len(gestos_treinados),
        'gestos': gestos_treinados,
        'webcam_index': selected_camera_index,
        'limiar_confianca': confidence_threshold
    })

# Rota de fallback para admin (simplificada)
@app.route('/admin')
@login_required
def admin_dashboard():
    if not current_user.is_admin():
        flash('Acesso restrito a administradores!', 'error')
        return redirect(url_for('camera_tradulibras'))
    
    user_stats = user_manager.get_stats()
    return render_template('admin_dashboard.html', user_stats=user_stats)

@app.route('/introducao')
@login_required
def introducao():
    return render_template('introducao.html', 
                         username=current_user.username, 
                         is_admin=current_user.is_admin())

@app.route('/tutorial')
@login_required
def tutorial():
    return render_template('tutorial.html')

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Página não encontrada"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Erro interno do servidor"}), 500

if __name__ == '__main__':
    print("🚀 TRADULIBRAS - SISTEMA INICIADO")
    print("=" * 50)
    print(f"📊 Gestos carregados: {len(gestos_treinados)}")
    print(f"🎯 Modelo: {'✅ Carregado' if model else '❌ Não carregado'}")
    print(f"🔮 Limiar de confiança: {confidence_threshold}")
    print(f"⏱️  Cooldown: {prediction_cooldown}s")
    print(f"📹 Webcam: {selected_camera_index}")
    print("🌐 Servidor: http://localhost:5000")
    print("=" * 50)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"❌ Erro ao iniciar servidor: {e}")