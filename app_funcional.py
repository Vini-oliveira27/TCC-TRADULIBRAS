#!/usr/bin/env python3
"""TraduLibras - Sistema de reconhecimento LIBRAS"""

from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import cv2, mediapipe as mp, numpy as np, pickle, os, tempfile, threading, time, glob
from gtts import gTTS
from datetime import datetime
from auth import user_manager, User

app = Flask(__name__)
app.secret_key = 'tradulibras_secret_key_2024'

# Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id): return user_manager.get_user(user_id)

# MediaPipe
mp_hands, mp_draw = mp.solutions.hands, mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Carregar modelo
pasta_modelos = 'modelos/'
modelos = sorted(glob.glob(os.path.join(pasta_modelos, 'modelo_libras.pkl')), key=os.path.getmtime)
scalers = sorted(glob.glob(os.path.join(pasta_modelos, 'scaler_libras.pkl')), key=os.path.getmtime)
infos = sorted(glob.glob(os.path.join(pasta_modelos, 'modelo_info.pkl')), key=os.path.getmtime)

if modelos and scalers and infos:
    modelo_inclusao_bc, scaler_inclusao_bc, info_inclusao_bc = modelos[-1], scalers[-1], infos[-1]
    with open(modelo_inclusao_bc, 'rb') as f: model = pickle.load(f)
    with open(scaler_inclusao_bc, 'rb') as f: scaler = pickle.load(f)
    with open(info_inclusao_bc, 'rb') as f: model_info = pickle.load(f)
    print(f"📊 Classes: {model_info['classes']}")
else: model, scaler, model_info = None, None, {'classes': [], 'accuracy': 0}

# Variáveis globais
current_letter = formed_text = ""
last_prediction_time, hand_detected_time = datetime.now(), None
prediction_cooldown, min_hand_time, auto_speak_enabled = 2.5, 1.5, True

def process_landmarks(hand_landmarks):
    if not hand_landmarks: return None
    wrist = hand_landmarks.landmark[0]
    features = [lm.x - wrist.x for lm in hand_landmarks.landmark] + [lm.y - wrist.y for lm in hand_landmarks.landmark]
    tips = [hand_landmarks.landmark[i] for i in [4,8,12,16,20]]
    features += [abs(tip.x - wrist.x) + abs(tip.y - wrist.y) for tip in tips]
    features += [abs(tips[i].x - tips[i+1].x) + abs(tips[i].y - tips[i+1].y) for i in range(4)]
    return features

def detectar_webcam_usb_automatico():
    for i in range(5):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened() and cap.read()[0]:
                cap.release()
                return i
        except: pass
    return 0

selected_camera_index = detectar_webcam_usb_automatico()

def generate_frames():
    global current_letter, formed_text, last_prediction_time, hand_detected_time, selected_camera_index
    camera = cv2.VideoCapture(selected_camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = camera.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        points, current_time = None, datetime.now()
        
        if results.multi_hand_landmarks:
            if hand_detected_time is None: hand_detected_time = current_time
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                points = process_landmarks(hand_landmarks)
            
            time_since_detection = (current_time - hand_detected_time).total_seconds()
            if time_since_detection >= min_hand_time:
                time_since_last = (current_time - last_prediction_time).total_seconds()
                if time_since_last >= prediction_cooldown and points and len(points) == 51:
                    try:
                        if model and scaler:
                            points_normalized = scaler.transform([points])
                            predicted_letter = model.predict(points_normalized)[0]
                            
                            if predicted_letter == 'ESPACO':
                                current_letter, formed_text = '[ESPAÇO]', formed_text + ' '
                            elif predicted_letter == '.':
                                current_letter = '[PONTO]'
                                texto_para_falar = formed_text.strip()
                                formed_text = ""
                                if texto_para_falar and auto_speak_enabled:
                                    threading.Thread(target=falar_texto_automatico, args=(texto_para_falar,), daemon=True).start()
                            else:
                                current_letter, formed_text = predicted_letter, formed_text + predicted_letter
                            
                            last_prediction_time, hand_detected_time = current_time, None
                    except Exception as e: print(f"❌ Erro: {e}")
        else: hand_detected_time, current_letter = None, ""
        
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    camera.release()

def falar_texto_automatico(texto_para_falar):
    try:
        if not texto_para_falar.strip(): return
        texto_limpo = texto_para_falar.strip()
        tts = gTTS(text=texto_limpo, lang='pt-br')
        temp_file = os.path.join(tempfile.gettempdir(), f'pygame_fala_{int(time.time())}.mp3')
        tts.save(temp_file)
        
        try:
            import pygame
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            start_time = time.time()
            while pygame.mixer.music.get_busy():
                if time.time() - start_time > 30: break
                time.sleep(0.1)
            pygame.mixer.quit()
        except: pass
        
        threading.Thread(target=lambda f: [time.sleep(10), os.path.exists(f) and os.remove(f)], args=(temp_file,)).start()
    except Exception as e: print(f"💥 ERRO: {e}")

# ==================== COMUNICAÇÃO SERIAL (MÃO ROBÓTICA) ====================
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

def diagnosticar_portas_seriais():
    if not SERIAL_AVAILABLE: return []
    try:
        ports = list(serial.tools.list_ports.comports())
        portas_detalhadas = []
        for port in ports:
            try:
                teste = serial.Serial(port.device)
                teste.close()
                status = "✅ Disponível"
            except: status = "❌ Indisponível"
            
            is_arduino = any(x in port.description.lower() for x in ['arduino', 'ch340', 'usb serial'])
            port_info = {
                'device': port.device, 'description': port.description,
                'hwid': port.hwid, 'is_arduino': is_arduino, 'status': status
            }
            portas_detalhadas.append(port_info)
        return portas_detalhadas
    except: return []

class SerialController:
    def __init__(self):
        self.serial_connection = None
        self.port = None
        self.baudrate = 115200
        self.connected = False
        
    def list_ports(self): return diagnosticar_portas_seriais()
    
    def connect(self, port):
        if not SERIAL_AVAILABLE: return False, "Biblioteca serial não disponível"
        try:
            self.serial_connection = serial.Serial(port=port, baudrate=self.baudrate, timeout=1, write_timeout=1)
            time.sleep(2)
            self.port = port
            self.connected = True
            return True, f"Conectado à porta {port}"
        except serial.SerialException as e:
            return False, f"Erro: {str(e)}"
    
    def disconnect(self):
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        self.connected = False
        self.port = None
        return True, "Desconectado"
    
    def send_letter(self, letter):
        if not self.connected or not self.serial_connection:
            return False, "Não conectado ao Arduino"
        try:
            letter = letter.lower().strip()
            if len(letter) == 1 and (letter.isalpha() or letter == '0'):
                self.serial_connection.write(letter.encode() + b'\n')
                self.serial_connection.flush()
                return True, f"Letra '{letter.upper()}' enviada"
            else: return False, "Letra inválida"
        except Exception as e: return False, f"Erro ao enviar: {str(e)}"
    
    def get_status(self):
        return {'connected': self.connected, 'port': self.port, 'serial_available': SERIAL_AVAILABLE}

serial_controller = SerialController()

# Rotas Serial
@app.route('/serial/ports')
@login_required
def get_serial_ports(): return jsonify({'ports': serial_controller.list_ports()})

@app.route('/serial/connect', methods=['POST'])
@login_required
def serial_connect():
    port = request.get_json().get('port')
    if not port: return jsonify({'success': False, 'message': 'Porta não especificada'})
    success, message = serial_controller.connect(port)
    return jsonify({'success': success, 'message': message})

@app.route('/serial/disconnect', methods=['POST'])
@login_required
def serial_disconnect():
    success, message = serial_controller.disconnect()
    return jsonify({'success': success, 'message': message})

@app.route('/serial/status')
@login_required
def serial_status(): return jsonify(serial_controller.get_status())

@app.route('/serial/send_letter', methods=['POST'])
@login_required
def send_serial_letter():
    letter = request.get_json().get('letter', '')
    if not letter: return jsonify({'success': False, 'message': 'Letra não especificada'})
    success, message = serial_controller.send_letter(letter)
    return jsonify({'success': success, 'message': message})

@app.route('/serial/send_word', methods=['POST'])
@login_required
def send_serial_word():
    word = request.get_json().get('word', '')
    if not word: return jsonify({'success': False, 'message': 'Palavra não especificada'})
    if not serial_controller.connected: return jsonify({'success': False, 'message': 'Não conectado ao Arduino'})
    
    results = []
    for letter in word.lower():
        if letter.isalpha() or letter == ' ':
            if letter == ' ': 
                time.sleep(1)
                results.append("Espaço - pausa")
            else:
                success, message = serial_controller.send_letter(letter)
                results.append(f"{letter.upper()}: {message}")
                time.sleep(0.8)
    return jsonify({'success': True, 'results': results})

# Rotas principais
@app.route('/')
def index(): return redirect(url_for('login' if not current_user.is_authenticated else 'introducao'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = user_manager.authenticate(request.form['username'], request.form['password'])
        if user: login_user(user); return redirect(url_for('introducao'))
        else: flash('Usuário ou senha incorretos!')
    return render_template('login.html')

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
    return render_template('introducao.html', username=current_user.username, is_admin=current_user.is_admin())

@app.route('/tutorial')
@login_required
def tutorial(): return render_template('tutorial.html')

@app.route('/logout') 
@login_required 
def logout(): logout_user(); flash('Desconectado.'); return redirect(url_for('login'))

@app.route('/camera') 
@login_required 
def camera_tradulibras(): return render_template('camera_tradulibras.html')

@app.route('/video_feed') 
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Rotas de controle
@app.route('/limpar_ultima_letra', methods=['POST'])
@login_required
def limpar_ultima_letra():
    global formed_text, current_letter
    if formed_text: formed_text = formed_text[:-1]; current_letter = ""
    return jsonify({"status": "success" if formed_text else "error", "texto": formed_text})

@app.route('/letra_atual') 
@login_required 
def get_letra_atual(): return jsonify({"letra": current_letter, "texto": formed_text})

@app.route('/limpar_texto', methods=['POST'])
@login_required 
def limpar_texto_completo(): global formed_text, current_letter; formed_text = current_letter = ""; return jsonify({"status": "success"})

@app.route('/falar_texto', methods=['GET', 'POST'])
@login_required
def falar_texto():
    if formed_text.strip():
        try:
            tts = gTTS(text=formed_text, lang='pt-br', slow=False)
            temp_file = os.path.join(tempfile.gettempdir(), f'manual_speech_{int(time.time())}.mp3')
            tts.save(temp_file)
            response = send_file(temp_file, mimetype='audio/mpeg', as_attachment=False)
            threading.Thread(target=lambda f: [time.sleep(30), os.path.exists(f) and os.remove(f)], args=(temp_file,)).start()
            return response
        except Exception as e: return jsonify({"success": False, "error": str(e)})
    return jsonify({"success": False, "error": "Texto vazio"})

@app.route('/auto_speak/toggle', methods=['POST'])
@login_required
def toggle_auto_speak():
    global auto_speak_enabled
    auto_speak_enabled = request.get_json().get('enabled', auto_speak_enabled)
    return jsonify({'success': True, 'auto_speak_enabled': auto_speak_enabled})

@app.route('/status')
@login_required
def status():
    return jsonify({
        "modelo_carregado": model is not None,
        "classes": model_info.get('classes', []),
        "acuracia": model_info.get('accuracy', 0),
        "texto_atual": formed_text,
        "letra_atual": current_letter
    })

if __name__ == '__main__':
    print("🚀 TRADULIBRAS - WEBCAM USB AUTOMÁTICA")
    print(f"📊 Classes: {model_info.get('classes', [])}")
    print(f"📹 Webcam: {selected_camera_index}")
    print("💡 Acesso: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)