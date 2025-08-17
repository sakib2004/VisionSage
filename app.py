from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from PIL import Image as PILImage
import sqlite3
import os
import io
import json
import re
import logging
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pytz
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
import torch
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.colors import HexColor, whitesmoke, black
import glob
import time

# Environment variable
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
if not app.secret_key:
    raise ValueError("No SECRET_KEY set in .env file")
app.jinja_env.filters['basename'] = os.path.basename
UPLOAD_FOLDER = 'Uploads'
REPORTS_FOLDER = 'Reports'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER

# Logging setup
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

# Create upload and report folders
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logging.info("Created Uploads folder")
if not os.path.exists(REPORTS_FOLDER):
    os.makedirs(REPORTS_FOLDER)
    logging.info("Created Reports folder")

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Timezone
tz = pytz.timezone('Asia/Dhaka')

# Pre-trained BLIP and DETR models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip_processor = None
blip_model = None
detr_processor = None
detr_model = None
model_load_error = None
try:
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16)
    blip_model.to(device)
    blip_model.eval()
    logging.info("Loaded pre-trained BLIP model")
    detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", torch_dtype=torch.float16, ignore_mismatched_sizes=True)
    detr_model.to(device)
    detr_model.eval()
    logging.info("Loaded pre-trained DETR model")
except Exception as e:
    logging.error(f"Failed to load models: {str(e)}")
    model_load_error = f"Image analysis models failed to load: {str(e)}. Some features may be unavailable."

# Image transformation for BLIP
def transform_image_blip(image):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224), PILImage.Resampling.LANCZOS)
        inputs = blip_processor(images=image, return_tensors="pt").to(device, torch.float16)
        return inputs
    except Exception as e:
        logging.error(f"BLIP image transformation failed: {str(e)}")
        raise

# Generate caption using BLIP
def generate_caption(image):
    try:
        if not blip_model or not blip_processor:
            return "BLIP model or processor not loaded."
        inputs = transform_image_blip(image)
        with torch.no_grad():
            generated_ids = blip_model.generate(**inputs, max_length=100, num_beams=5, early_stopping=True)
            caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        torch.cuda.empty_cache()
        return caption
    except Exception as e:
        logging.error(f"Caption generation failed: {str(e)}")
        return f"Caption generation error: {str(e)}"

# Description from caption
def get_short_description(caption):
    if len(caption) <= 50:
        return caption
    first_sentence = re.match(r'^.*?[.!?]', caption)
    return first_sentence.group(0) if first_sentence else caption[:50] + "..."

# Object detection with DETR
def detect_objects(image):
    try:
        if not detr_model or not detr_processor:
            return ["DETR model or processor not loaded."]
        inputs = detr_processor(images=image, return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            outputs = detr_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        results = detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
        objects = [f"{detr_model.config.id2label[label.item()]} ({score.item():.2f})" for score, label in zip(results["scores"], results["labels"])]
        torch.cuda.empty_cache()
        return objects[:3] if objects else ["No objects detected with high confidence"]
    except Exception as e:
        logging.error(f"Object detection failed: {str(e)}")
        return [f"Object detection error: {str(e)}"]

# Harmful objects with COCO names
HARMFUL_OBJECT_MAPPING = {
    'knife': ['knife', 'blade', 'dagger', 'sword', 'knives'],
    'gun': ['gun', 'firearm', 'pistol', 'rifle', 'revolver'],
    'weapon': ['knife', 'gun', 'firearm', 'pistol', 'rifle', 'sword', 'dagger', 'blade'],
    'blood': ['blood'],
    'syringe': ['syringe'],
    'bomb': ['bomb'],
    'blade': ['knife', 'blade', 'sword', 'dagger'],
    'dagger': ['dagger', 'knife', 'blade'],
    'sword': ['sword', 'blade'],
    'firearm': ['gun', 'firearm', 'pistol', 'rifle', 'revolver'],
    'pistol': ['pistol', 'gun', 'firearm', 'revolver']
}

# Metadata extraction
def extract_metadata(image_data, filename):
    try:
        image = PILImage.open(io.BytesIO(image_data))
        width, height = image.size
        file_size = len(image_data)
        metadata = {
            "Filename": filename,
            "Dimensions": f"{width}x{height}",
            "File Size": f"{file_size / 1024:.2f} KB"
        }
        try:
            from exif import Image as ExifImage
            exif_img = ExifImage(io.BytesIO(image_data))
            if exif_img.has_exif:
                metadata.update({
                    "Timestamp": exif_img.get("datetime_original", "N/A"),
                    "GPS": {
                        "Latitude": exif_img.get("gps_latitude", "N/A"),
                        "Longitude": exif_img.get("gps_longitude", "N/A")
                    },
                    "Camera": f"{exif_img.get('make', 'N/A')} {exif_img.get('model', 'N/A')}"
                })
        except Exception as e:
            logging.warning(f"EXIF metadata extraction failed: {str(e)}")
        return metadata
    except Exception as e:
        logging.error(f"Metadata extraction failed: {str(e)}")
        return {"Error": f"Metadata extraction failed: {str(e)}"}

# Image compression
def compress_image(image_data, max_size=(1024, 1024), quality=85):
    try:
        img = PILImage.open(io.BytesIO(image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.thumbnail(max_size, PILImage.Resampling.LANCZOS)
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=quality)
        logging.debug("Image compressed successfully")
        return output.getvalue()
    except Exception as e:
        logging.error(f"Image compression failed: {str(e)}")
        return image_data

# Database connection
def get_db_connection():
    try:
        conn = sqlite3.connect('investigations.db', timeout=30)
        conn.row_factory = sqlite3.Row
        logging.debug("Database connection established")
        return conn
    except Exception as e:
        logging.error(f"Database connection failed: {str(e)}")
        raise

# Database initialization
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''CREATE TABLE IF NOT EXISTS users
                          (id INTEGER PRIMARY KEY AUTOINCREMENT,
                           name TEXT NOT NULL,
                           phone TEXT NOT NULL UNIQUE,
                           password TEXT NOT NULL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS cases
                          (id INTEGER PRIMARY KEY AUTOINCREMENT,
                           case_name TEXT NOT NULL,
                           image_id TEXT NOT NULL,
                           analysis_result TEXT,
                           timestamp TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS images
                          (id INTEGER PRIMARY KEY AUTOINCREMENT,
                           file_path TEXT NOT NULL,
                           description TEXT,
                           harmful_objects TEXT,
                           detected_objects TEXT,
                           metadata TEXT,
                           case_id INTEGER,
                           user_id INTEGER,
                           timestamp TEXT,
                           FOREIGN KEY(user_id) REFERENCES users(id),
                           FOREIGN KEY(case_id) REFERENCES cases(id))''')
        conn.commit()
        logging.info("Database initialized successfully")
    except Exception as e:
        logging.error(f"Database initialization failed: {str(e)}")
        raise
    finally:
        conn.close()
        logging.debug("Database connection closed in init_db")

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id, name, phone):
        self.id = id
        self.name = name
        self.phone = phone

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cursor = conn.execute("SELECT id, name, phone FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return User(user['id'], user['name'], user['phone'])
    return None

# File validation
def allowed_file(filename):
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def validate_image_size(image_data):
    return len(image_data) <= 10 * 1024 * 1024

def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9._-]', '_', filename)

# Index route
@app.route('/')
def index():
    if model_load_error and current_user.is_authenticated:
        flash(model_load_error, 'error')
    try:
        conn = get_db_connection()
        cursor = conn.execute("SELECT id, file_path, description, harmful_objects, detected_objects, metadata, case_id, timestamp FROM images WHERE user_id = ?",
                             (current_user.id,) if current_user.is_authenticated else (0,))
        images = cursor.fetchall()
        conn.close()
        if not images:
            logging.debug("No images found for user_id: {}".format(current_user.id if current_user.is_authenticated else 0))
            return render_template('index.html', images=[])
        images_list = []
        for img in images:
            try:
                case_id = None
                if img['case_id'] is not None:
                    try:
                        case_id = int(img['case_id'])
                    except ValueError:
                        logging.warning(f"Invalid case_id {img['case_id']} for image {img['id']}")
                images_list.append({
                    'id': img['id'],
                    'file_path': img['file_path'],
                    'description': img['description'],
                    'harmful_objects': img['harmful_objects'],
                    'detected_objects': img['detected_objects'],
                    'metadata': json.loads(img['metadata']) if img['metadata'] else None,
                    'case_id': case_id,
                    'timestamp': img['timestamp']
                })
            except Exception as e:
                logging.error(f"Error processing image record {img['id']}: {str(e)}")
                continue
        if not images_list:
            logging.warning("No valid images after processing")
            flash('No valid images found.', 'warning')
        return render_template('index.html', images=images_list)
    except Exception as e:
        logging.error(f"Error in index route: {str(e)}")
        flash(f'Error loading images: {str(e)}', 'error')
        return render_template('index.html', images=[])

# Other routes
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        phone = request.form['phone']
        password = request.form['password']
        try:
            conn = get_db_connection()
            cursor = conn.execute("SELECT id, name, phone, password FROM users WHERE phone = ?", (phone,))
            user = cursor.fetchone()
            conn.close()
            if user and check_password_hash(user['password'], password):
                login_user(User(user['id'], user['name'], user['phone']))
                flash('Login successful!', 'success')
                return redirect(url_for('index'))
            flash('Invalid phone or password.', 'error')
        except Exception as e:
            logging.error(f"Login error: {str(e)}")
            flash(f'Login error: {str(e)}', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        name = request.form['name']
        phone = request.form['phone']
        password = request.form['password']
        if not (name and phone and password):
            flash('All fields are required.', 'error')
            return render_template('register.html')
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id FROM users WHERE phone = ?", (phone,))
            if cursor.fetchone():
                conn.close()
                flash('Phone number already registered.', 'error')
                return render_template('register.html')
            hashed_password = generate_password_hash(password)
            cursor.execute("INSERT INTO users (name, phone, password) VALUES (?, ?, ?)",
                          (name, phone, hashed_password))
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            login_user(User(user_id, name, phone))
            flash('Registration successful!', 'success')
            return redirect(url_for('index'))
        except sqlite3.OperationalError as e:
            conn.close()
            logging.error(f"Register database error: {str(e)}")
            flash(f'Database error: {str(e)}', 'error')
            return render_template('register.html')
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
@login_required
def upload_image():
    logging.debug("Starting image upload process")
    images = [request.files.get(f'image_{i}') for i in range(1, 4)]
    images = [img for img in images if img and img.filename]
    if not images:
        logging.warning("No images uploaded")
        flash('No images uploaded.', 'error')
        return redirect(url_for('index'))
    if len(images) > 3:
        logging.warning("Too many images uploaded")
        flash('Maximum 3 images allowed.', 'error')
        return redirect(url_for('index'))

    analysis_options = {
        'description': request.form.get('description') == 'true',
        'harmful': request.form.get('harmful') == 'true',
        'detected_objects': request.form.get('detected_objects') == 'true',
        'metadata': request.form.get('metadata') == 'true',
        'case_folder': request.form.get('case_folder', '').strip() or 'Default',
        'case_action': request.form.get('case_action', 'add')
    }
    if not any([analysis_options['description'], analysis_options['harmful'], analysis_options['detected_objects'], analysis_options['metadata']]):
        logging.warning("No analysis options selected")
        flash('At least one analysis option must be selected.', 'error')
        return redirect(url_for('index'))

    custom_harmful = request.form.get('custom_harmful', '').lower().split(',')
    custom_harmful = [obj.strip() for obj in custom_harmful if obj.strip()]
    harmful_objects = ['knife', 'gun', 'weapon', 'blood', 'syringe', 'bomb', 'blade', 'dagger', 'sword', 'firearm', 'pistol'] + custom_harmful

    results = []
    timestamp = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        case_name = analysis_options['case_folder']
        cursor.execute("SELECT id FROM cases WHERE case_name = ?", (case_name,))
        case = cursor.fetchone()
        if not case:
            cursor.execute("INSERT INTO cases (case_name, image_id, analysis_result, timestamp) VALUES (?, ?, ?, ?)",
                          (case_name, '', '', timestamp))
            case_id = cursor.lastrowid
        else:
            case_id = case['id']

        for idx, image in enumerate(images, 1):
            logging.debug(f"Processing image {idx}: {image.filename}")
            if not allowed_file(image.filename):
                logging.warning(f"Image_{idx}: Invalid file format - {image.filename}")
                flash(f'Image_{idx}: Invalid file format. Use JPG or PNG.', 'error')
                continue
            image_data = image.read()
            if not validate_image_size(image_data):
                logging.warning(f"Image_{idx}: File size exceeds 10MB - {image.filename}")
                flash(f'Image_{idx}: File size exceeds 10MB.', 'error')
                continue

            # Compress image
            image_data = compress_image(image_data)
            sanitized_filename = sanitize_filename(f"{current_user.id}_{timestamp.replace(':', '-')}_{image.filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], sanitized_filename)
            logging.debug(f"Saving image to {file_path}")
            with open(file_path, 'wb') as f:
                f.write(image_data)

            result = {'image_id': f'Image_{idx}', 'timestamp': timestamp}
            image.seek(0)
            try:
                pil_image = PILImage.open(io.BytesIO(image_data))
            except Exception as e:
                logging.error(f"Image_{idx}: Failed to open image - {str(e)}")
                flash(f'Image_{idx}: Invalid image file - {str(e)}', 'error')
                continue

            # Generate caption for description and harmful object fallback
            caption = None
            if analysis_options['description'] or analysis_options['harmful']:
                caption = generate_caption(pil_image)
                logging.debug(f"Image_{idx}: Caption generated: {caption}")

            if analysis_options['description']:
                result['description'] = get_short_description(caption) if caption else "Caption generation failed."
                logging.debug(f"Image_{idx}: Description generated: {result['description']}")

            if analysis_options['detected_objects']:
                try:
                    detected_objects = detect_objects(pil_image)
                    result['detected_objects'] = ", ".join(detected_objects) if detected_objects else "No objects detected."
                    logging.debug(f"Image_{idx}: DETR detected objects: {result['detected_objects']}")
                except Exception as e:
                    logging.error(f"Image_{idx}: Object detection error - {str(e)}")
                    result['detected_objects'] = f"Object detection error: {str(e)}"

            if analysis_options['harmful']:
                try:
                    detected_objects = detect_objects(pil_image)
                    logging.debug(f"Image_{idx}: DETR detected objects for harmful analysis: {detected_objects}")
                    harmful_detected = []
                    for obj in detected_objects:
                        obj_lower = obj.lower()
                        for harmful_key, harmful_aliases in HARMFUL_OBJECT_MAPPING.items():
                            if any(alias.lower() in obj_lower for alias in harmful_aliases):
                                harmful_detected.append(f"{harmful_key} ({obj})")
                                break
                        for custom in custom_harmful:
                            if custom.lower() in obj_lower:
                                harmful_detected.append(f"{custom} ({obj})")
                                break
                    if not harmful_detected and caption:
                        for harmful_key, harmful_aliases in HARMFUL_OBJECT_MAPPING.items():
                            if any(alias.lower() in caption.lower() for alias in harmful_aliases):
                                harmful_detected.append(f"{harmful_key} (detected in caption)")
                        for custom in custom_harmful:
                            if custom.lower() in caption.lower():
                                harmful_detected.append(f"{custom} (detected in caption)")
                    result['harmful_objects'] = f"Detected: {', '.join(harmful_detected)}" if harmful_detected else "No harmful objects detected."
                    logging.debug(f"Image_{idx}: Harmful objects analyzed: {result['harmful_objects']}")
                except Exception as e:
                    logging.error(f"Image_{idx}: Harmful objects error - {str(e)}")
                    result['harmful_objects'] = f"Harmful objects error: {str(e)}"

            if analysis_options['metadata']:
                result['metadata'] = extract_metadata(image_data, sanitized_filename)
                logging.debug(f"Image_{idx}: Metadata extracted: {result['metadata']}")

            result['case_id'] = case_id
            result['user_id'] = current_user.id

            try:
                cursor.execute("INSERT INTO images (file_path, description, harmful_objects, detected_objects, metadata, case_id, user_id, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                              (file_path, result.get('description'), result.get('harmful_objects'), result.get('detected_objects'),
                               json.dumps(result.get('metadata')) if result.get('metadata') else None,
                               result['case_id'], result['user_id'], timestamp))
                image_id = cursor.lastrowid
                cursor.execute("UPDATE cases SET image_id = ?, analysis_result = ? WHERE id = ?",
                              (f"{image_id}", json.dumps(result), case_id))
                logging.debug(f"Image_{idx}: Database insertion successful")
                results.append(result)
            except Exception as e:
                logging.error(f"Image_{idx}: Database insertion failed - {str(e)}")
                flash(f'Image_{idx}: Database error - {str(e)}', 'error')

        conn.commit()
        logging.info("Database commit successful")
    except Exception as e:
        logging.error(f"Upload route error: {str(e)}")
        flash(f'Upload error: {str(e)}', 'error')
    finally:
        if conn:
            conn.close()
            logging.debug("Database connection closed")

    if results:
        flash('Images processed successfully.', 'success')
    return redirect(url_for('index'))

@app.route('/Uploads/<filename>')
def uploaded_file(filename):
    try:
        logging.debug(f"Serving file: {filename}")
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        logging.error(f"Error serving file {filename}: {str(e)}")
        flash(f'Error serving image: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        conn = get_db_connection()
        cursor = conn.execute("SELECT COUNT(*) as img_count FROM images WHERE user_id = ?", (current_user.id,))
        img_count = cursor.fetchone()['img_count']
        cursor = conn.execute("SELECT COUNT(*) as case_count FROM cases WHERE id IN (SELECT case_id FROM images WHERE user_id = ?)", (current_user.id,))
        case_count = cursor.fetchone()['case_count']
        cursor = conn.execute("SELECT harmful_objects FROM images WHERE user_id = ? AND harmful_objects IS NOT NULL", (current_user.id,))
        harmful_count = sum(1 for row in cursor.fetchall() if row['harmful_objects'] != "No harmful objects detected.")
        conn.close()
        stats = {'images': img_count, 'cases': case_count, 'harmful': harmful_count}
        return render_template('dashboard.html', stats=stats)
    except Exception as e:
        logging.error(f"Dashboard error: {str(e)}")
        flash(f'Error loading dashboard: {str(e)}', 'error')
        return render_template('dashboard.html', stats={'images': 0, 'cases': 0, 'harmful': 0})

@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    try:
        if request.method == 'POST':
            keyword = request.form.get('keyword', '')
            date_from = request.form.get('date_from', '')
            date_to = request.form.get('date_to', '')
            query = "SELECT id, file_path, description, harmful_objects, detected_objects, metadata, case_id, timestamp FROM images WHERE user_id = ?"
            params = [current_user.id]
            if keyword:
                query += " AND (description LIKE ? OR harmful_objects LIKE ? OR detected_objects LIKE ?)"
                params.extend([f'%{keyword}%', f'%{keyword}%', f'%{keyword}%'])
            if date_from:
                query += " AND timestamp >= ?"
                params.append(date_from)
            if date_to:
                query += " AND timestamp <= ?"
                params.append(date_to)
            conn = get_db_connection()
            cursor = conn.execute(query, params)
            images = cursor.fetchall()
            conn.close()
            images_list = []
            for img in images:
                try:
                    case_id = None
                    if img['case_id'] is not None:
                        try:
                            case_id = int(img['case_id'])
                        except ValueError:
                            logging.warning(f"Invalid case_id {img['case_id']} for image {img['id']}")
                    images_list.append({
                        'id': img['id'],
                        'file_path': img['file_path'],
                        'description': img['description'],
                        'harmful_objects': img['harmful_objects'],
                        'detected_objects': img['detected_objects'],
                        'metadata': json.loads(img['metadata']) if img['metadata'] else None,
                        'case_id': case_id,
                        'timestamp': img['timestamp']
                    })
                except Exception as e:
                    logging.error(f"Error processing search image record {img['id']}: {str(e)}")
                    continue
            return render_template('search.html', images=images_list, keyword=keyword, date_from=date_from, date_to=date_to)
        return render_template('search.html', images=[])
    except Exception as e:
        logging.error(f"Search error: {str(e)}")
        flash(f'Error searching images: {str(e)}', 'error')
        return render_template('search.html', images=[])

@app.route('/case_timeline/<int:case_id>')
@login_required
def case_timeline(case_id):
    try:
        conn = get_db_connection()
        cursor = conn.execute("SELECT timestamp, description, file_path FROM images WHERE case_id = ? ORDER BY timestamp", (case_id,))
        events = [{'timestamp': row['timestamp'], 'description': row['description'] or 'Image uploaded', 'file_path': row['file_path']} for row in cursor.fetchall()]
        cursor = conn.execute("SELECT case_name FROM cases WHERE id = ?", (case_id,))
        case = cursor.fetchone()
        case_name = case['case_name'] if case else f"Case {case_id}"
        conn.close()
        return render_template('timeline.html', events=events, case_id=case_id, case_name=case_name)
    except Exception as e:
        logging.error(f"Timeline error: {str(e)}")
        flash(f'Error loading timeline: {str(e)}', 'error')
        return render_template('timeline.html', events=[], case_id=case_id, case_name=f"Case {case_id}")
    

@app.route('/print_user_data')
@login_required
def print_user_data():
    try:
        # Clean up old PDFs (older than 7 days)
        for old_file in glob.glob(os.path.join(app.config['REPORTS_FOLDER'], f"user_data_{current_user.id}_*.pdf")):
            if os.path.getmtime(old_file) < time.time() - 7 * 24 * 3600:
                os.remove(old_file)
                logging.info(f"Deleted old PDF: {old_file}")

        # Fetch user data
        conn = get_db_connection()
        cursor = conn.execute("SELECT id, name, phone FROM users WHERE id = ?", (current_user.id,))
        user = cursor.fetchone()
       
        # Fetch user's images (limited to last 100)
        cursor = conn.execute("SELECT id, file_path, description, harmful_objects, detected_objects, metadata, case_id, timestamp FROM images WHERE user_id = ? ORDER BY timestamp DESC LIMIT 100",
                             (current_user.id,))
        images = cursor.fetchall()
       
        # Fetch user's cases
        cursor = conn.execute("SELECT id, case_name, image_id, analysis_result, timestamp FROM cases WHERE id IN (SELECT case_id FROM images WHERE user_id = ?)",
                             (current_user.id,))
        cases = cursor.fetchall()
        conn.close()

        # Prepare PDF
        filename = sanitize_filename(f"user_data_{current_user.id}_{datetime.now(tz).strftime('%Y%m%d_%H%M%S')}.pdf")
        filepath = os.path.join(app.config['REPORTS_FOLDER'], filename)
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        normal_style = styles['Normal']
        normal_style.fontSize = 9
        normal_style.leading = 11
        elements = []

        # Title
        elements.append(Paragraph("VISIONSAGE", styles['Title']))
        elements.append(Spacer(1, 12))

        # User Details
        elements.append(Paragraph("User Details", styles['Heading2']))
        user_data = [
            ["User ID", Paragraph(str(user['id']), normal_style)],
            ["Name", Paragraph(user['name'], normal_style)],
            ["Phone", Paragraph(user['phone'], normal_style)]
        ]
        user_table = Table(user_data, colWidths=[100, 400])
        user_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#FF6F61')),
            ('TEXTCOLOR', (0, 0), (-1, 0), whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#FFD166')),
            ('GRID', (0, 0), (-1, -1), 1, black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6)
        ]))
        elements.append(user_table)
        elements.append(Spacer(1, 12))

        # Images
        elements.append(Paragraph("Images", styles['Heading2']))
        if images:
            image_data = [["ID", "Image", "Description", "Harmful Objects", "Detected Objects", "Case ID", "Timestamp"]]
            for img in images:
                metadata = json.loads(img['metadata']) if img['metadata'] else {}
                metadata_str = "; ".join([f"{k}: {v}" for k, v in metadata.items()])
                image_path = img['file_path']
                try:
                    img_obj = ReportLabImage(image_path, width=50, height=50)
                    img_obj.hAlign = 'CENTER'
                except Exception as e:
                    logging.warning(f"Failed to load image {image_path} for PDF: {str(e)}")
                    img_obj = Paragraph("Image N/A", normal_style)
                image_data.append([
                    Paragraph(str(img['id']), normal_style),
                    img_obj,
                    Paragraph(img['description'] or "N/A", normal_style),
                    Paragraph(img['harmful_objects'] or "N/A", normal_style),
                    Paragraph(img['detected_objects'] or "N/A", normal_style),
                    Paragraph(str(img['case_id']) if img['case_id'] else "N/A", normal_style),
                    Paragraph(img['timestamp'], normal_style)
                ])
            row_heights = [30] + [60 for _ in range(len(image_data) - 1)]
            image_table = Table(image_data, colWidths=[30, 60, 120, 120, 120, 50, 100], rowHeights=row_heights)
            image_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#FF6F61')),
                ('TEXTCOLOR', (0, 0), (-1, 0), whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), HexColor('#FFD166')),
                ('GRID', (0, 0), (-1, -1), 1, black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TEXTWRAP', (0, 1), (-1, -1), 1)
            ]))
            elements.append(image_table)
        else:
            elements.append(Paragraph("No images found.", normal_style))
        elements.append(Spacer(1, 12))

        # Cases
        elements.append(Paragraph("Cases", styles['Heading2']))
        if cases:
            case_data = [["ID", "Case Name", "Image ID", "Analysis Result", "Timestamp"]]
            for case in cases:
                case_data.append([
                    Paragraph(str(case['id']), normal_style),
                    Paragraph(case['case_name'], normal_style),
                    Paragraph(case['image_id'], normal_style),
                    Paragraph(case['analysis_result'] or "N/A", normal_style),
                    Paragraph(case['timestamp'], normal_style)
                ])
            case_table = Table(case_data, colWidths=[50, 150, 100, 150, 100])
            case_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#FF6F61')),
                ('TEXTCOLOR', (0, 0), (-1, 0), whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), HexColor('#FFD166')),
                ('GRID', (0, 0), (-1, -1), 1, black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TEXTWRAP', (0, 1), (-1, -1), 1)
            ]))
            elements.append(case_table)
        else:
            elements.append(Paragraph("No cases found.", normal_style))

        # Build PDF
        doc.build(elements)
        logging.info(f"Generated PDF report: {filename}")
        return send_from_directory(app.config['REPORTS_FOLDER'], filename, as_attachment=True)
    except Exception as e:
        logging.error(f"Error generating PDF: {str(e)}")
        flash(f"Error generating PDF: {str(e)}", 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5002, debug=True)