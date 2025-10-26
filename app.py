import os
import sqlite3
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from gradcam import GradCAM  # Ensure gradcam.py is present

app = Flask(__name__)
app.secret_key = 'your_actual_secret_key'  # Use ENV var in production
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
MODEL_PATH = "resunitnet_glioma.pth"

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    # Only email and password now
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
init_db()

class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class ResUnitNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ResUnitNet, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.layer1 = ResUnit(32, 32)
        self.layer2 = ResUnit(32, 64, stride=2)
        self.layer3 = ResUnit(64, 128, stride=2)
        self.layer4 = ResUnit(128, 256, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResUnitNet(num_classes=len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def get_prediction(image_tensor):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()], predicted.item(), probs[0][predicted.item()].item()

def calculate_tumor_percentage(heatmap):
    threshold = 0.5
    tumor_area = np.sum(heatmap > threshold)
    total_area = heatmap.size
    percentage = (tumor_area / total_area) * 100
    return round(percentage, 2)

def estimate_tumor_stage(percentage):
    if percentage < 10:
        return "Stage I (Low)"
    elif percentage < 30:
        return "Stage II (Moderate)"
    elif percentage < 60:
        return "Stage III (High)"
    else:
        return "Stage IV (Critical)"

@app.route('/', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('upload_file'))
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('SELECT password FROM users WHERE email = ?', (email,))
            result = c.fetchone()
            conn.close()
            if result and check_password_hash(result[0], password):
                session['user'] = email
                flash('Login successful!', 'success')
                return redirect(url_for('upload_file'))
            else:
                error = 'Invalid email or password'
        except Exception as e:
            error = 'An error occurred'
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user' in session:
        return redirect(url_for('upload_file'))
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        hashed_pw = generate_password_hash(password)
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('INSERT INTO users (email, password) VALUES (?, ?)', (email, hashed_pw))
            conn.commit()
            conn.close()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error='Email already exists')
    return render_template('register.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('index.html', prediction=None, error='Please select a file to upload.')
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                img_tensor = transform_image(filepath)
                pred, pred_idx, pred_prob = get_prediction(img_tensor)
                if pred != 'notumor':
                    cam = GradCAM(model, model.layer4)
                    heatmap = cam.generate(img_tensor.to(device), class_idx=pred_idx)
                    img_cv = cv2.imread(filepath)
                    img_cv = cv2.resize(img_cv, (224, 224))
                    heatmap_color = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
                    overlayed = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
                    result_filename = 'result_' + filename
                    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                    cv2.imwrite(result_path, overlayed)
                    tumor_percentage = calculate_tumor_percentage(heatmap)
                    tumor_stage = estimate_tumor_stage(tumor_percentage)
                else:
                    result_filename = None
                    tumor_percentage = None
                    tumor_stage = None
                return render_template('index.html',
                                       filename=filename,
                                       prediction=pred,
                                       result_image=result_filename,
                                       tumor_percentage=tumor_percentage,
                                       tumor_stage=tumor_stage,
                                       pred_prob=round(pred_prob * 100, 2))
            except Exception as e:
                return render_template('index.html', prediction=None, error=f'Error during prediction: {str(e)}')
        else:
            return render_template('index.html', prediction=None, error='Invalid file type. Only PNG, JPG, JPEG allowed.')
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)


