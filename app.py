import os
import json
from datetime import datetime
from functools import wraps
from pathlib import Path
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file

import whitesox_analysis
from cubs import cubs_bp

app = Flask(__name__)
app.jinja_env.globals['enumerate'] = enumerate
app.secret_key = os.environ.get('SECRET_KEY', 'dev-only-insecure-key')

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
DATA_FOLDER = 'data'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# File paths for data storage
PROFILE_FILE = os.path.join(DATA_FOLDER, 'profile.json')
POSTS_FILE = os.path.join(DATA_FOLDER, 'posts.json')
CONFIG_FILE = os.path.join(DATA_FOLDER, 'config.json')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

app.register_blueprint(cubs_bp)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_json(filepath, default=None):
    """Load JSON file, return default if not found."""
    if not os.path.exists(filepath):
        return default if default is not None else {}
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return default if default is not None else {}


def save_json(filepath, data):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_profile():
    """Load profile data."""
    default_profile = {
        "name": "Kyle Flynn",
        "subtitle": "Baseball Operations & Finance",
        "bio": "Welcome to my portfolio! I'm passionate about baseball operations, financial analysis, and data-driven decision-making.",
        "photo": None
    }
    return load_json(PROFILE_FILE, default_profile)


def load_posts():
    """Load all posts."""
    data = load_json(POSTS_FILE, [])
    baseball_keywords = ['baseball', 'sabermetric', 'war', 'yankee', 'sox', 'cub', 'lineup', 'woba', 'obp', 'era', 'mlb']
    for post in data:
        if 'id' not in post:
            post['id'] = str(datetime.now().timestamp())
        if 'tags' not in post:
            post['tags'] = []
        if 'category' not in post:
            tags_str = ' '.join(post.get('tags', [])).lower()
            post['category'] = 'baseball' if any(kw in tags_str for kw in baseball_keywords) else 'personal'
    return data


def load_admin_password():
    """Load admin password hash from config."""
    config = load_json(CONFIG_FILE, {})
    return config.get('admin_password_hash')


def get_next_post_id():
    """Generate next post ID."""
    posts = load_posts()
    if not posts:
        return 1
    return max(int(p.get('id', 0)) for p in posts if str(p.get('id', 0)).isdigit()) + 1


def login_required(f):
    """Decorator to require admin login."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin' not in session:
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/')
def index():
    """Home page."""
    profile = load_profile()
    return render_template('index.html', profile=profile)


@app.route('/blog')
def blog():
    """Blog/projects page."""
    posts = load_posts()
    # Sort by date descending
    posts.sort(key=lambda x: x.get('date', ''), reverse=True)
    return render_template('blog.html', posts=posts)


@app.route('/connect')
def connect():
    """Connect/contact page."""
    return render_template('connect.html')


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page."""
    if request.method == 'POST':
        password = request.form.get('password', '')
        password_hash = load_admin_password()
        
        if password_hash and check_password_hash(password_hash, password):
            session['admin'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            error = "Invalid password"
            return render_template('admin_login.html', error=error)
    
    return render_template('admin_login.html')


@app.route('/admin')
@login_required
def admin_dashboard():
    """Admin dashboard."""
    profile = load_profile()
    posts = load_posts()
    posts.sort(key=lambda x: x.get('date', ''), reverse=True)
    return render_template('admin_dashboard.html', profile=profile, posts=posts)


@app.route('/admin/save_profile', methods=['POST'])
@login_required
def save_profile():
    """Save profile changes."""
    profile = load_profile()
    profile['name'] = request.form.get('name', profile['name'])
    profile['subtitle'] = request.form.get('subtitle', profile['subtitle'])
    profile['bio'] = request.form.get('bio', profile['bio'])
    
    # Handle photo upload
    if 'photo' in request.files:
        file = request.files['photo']
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(f"profile_{datetime.now().timestamp()}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            profile['photo'] = f"/static/uploads/{filename}"
    
    save_json(PROFILE_FILE, profile)
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/add_post', methods=['POST'])
@login_required
def add_post():
    """Add new blog post."""
    posts = load_posts()
    
    # Create new post
    date_input = request.form.get('date', '')
    if date_input:
        # Convert YYYY-MM-DD to "Month DD, YYYY"
        date_obj = datetime.strptime(date_input, '%Y-%m-%d')
        formatted_date = date_obj.strftime('%B %d, %Y')
    else:
        formatted_date = datetime.now().strftime('%B %d, %Y')
    
    new_post = {
        'id': str(get_next_post_id()),
        'title': request.form.get('title', ''),
        'description': request.form.get('description', ''),
        'date': formatted_date,
        'project_url': request.form.get('project_url', '') or None,
        'github_url': request.form.get('github_url', '') or None,
        'tags': [tag.strip() for tag in request.form.get('tags', '').split(',') if tag.strip()],
        'category': request.form.get('category', 'baseball'),
        'image': None
    }
    
    # Handle image upload
    if 'image' in request.files:
        file = request.files['image']
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(f"post_{new_post['id']}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            new_post['image'] = f"/static/uploads/{filename}"
    
    posts.append(new_post)
    save_json(POSTS_FILE, posts)
    
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/delete_post/<post_id>', methods=['POST'])
@login_required
def delete_post(post_id):
    """Delete a blog post."""
    posts = load_posts()
    posts = [p for p in posts if str(p.get('id')) != post_id]
    save_json(POSTS_FILE, posts)
    
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/logout')
def admin_logout():
    """Logout admin."""
    session.pop('admin', None)
    return redirect(url_for('index'))


@app.route('/admin/setup', methods=['GET', 'POST'])
def admin_setup():
    """Initial admin password setup."""
    config = load_json(CONFIG_FILE, {})
    
    if config.get('admin_password_hash'):
        return redirect(url_for('admin_login'))
    
    if request.method == 'POST':
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        if not password or len(password) < 6:
            error = "Password must be at least 6 characters"
            return render_template('admin_setup.html', error=error)
        
        if password != confirm_password:
            error = "Passwords do not match"
            return render_template('admin_setup.html', error=error)
        
        config['admin_password_hash'] = generate_password_hash(password)
        save_json(CONFIG_FILE, config)
        
        session['admin'] = True
        return redirect(url_for('admin_dashboard'))
    
    return render_template('admin_setup.html')


@app.route('/projects/jordan-walker-2026')
def jordan_walker_2026():
    """Jordan Walker 2026 breakout analysis."""
    return render_template('projects/jordan_walker_2026.html')


@app.route('/projects/yankees-baseball-operations')
def yankees_baseball_operations():
    """Yankees Baseball Operations Portfolio — 2026-27 analysis."""
    return render_template('projects/yankees_baseball_ops.html')


@app.route('/projects/whitesox-2026')
def whitesox_2026():
    """Rebuilding the 2026 Chicago White Sox — WAR-based valuation model."""
    vals = whitesox_analysis.build_valuations()
    return render_template(
        'projects/whitesox_2026.html',
        valuations=vals,
        summary=whitesox_analysis.valuation_summary(vals),
        revenue=whitesox_analysis.revenue_model(vals),
        playing_time=whitesox_analysis.playing_time_analysis(vals),
        saber_lineup=whitesox_analysis.sabermetric_lineup(),
        injured=whitesox_analysis.injured_list(),
    )


@app.route('/resume')
def resume():
    """Professional resume page."""
    return render_template('resume.html')


@app.route('/download-resume')
def download_resume():
    """Download resume file."""
    resume_path = os.path.join(app.root_path, 'static', 'resume', 'Kyle_Flynn_Resume.pdf')

    if os.path.exists(resume_path):
        return send_file(
            resume_path,
            as_attachment=True,
            download_name='Kyle_Flynn_Resume.pdf'
        )
    else:
        return redirect(url_for('resume'))


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
