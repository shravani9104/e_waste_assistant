from flask import Flask, request, render_template, render_template_string, session, redirect, url_for, flash, jsonify
from decision_engine import decide_action, get_questions, get_recommendations, get_eco_score, get_environmental_impact, get_educational_tips
from model import detect_device
from database import Database
from werkzeug.utils import secure_filename
import os
from functools import wraps

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
db = Database()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('signin'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('email') != 'admin@ewasteassistant.com':
            flash('Admin access required', 'error')
            return redirect(url_for('signin'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    return render_template('index.html', page='home')

@app.route('/about')
def about():
    return render_template('index.html', page='about')

@app.route('/contact')
def contact():
    return render_template('index.html', page='contact')

@app.route('/help')
def help():
    return render_template('index.html', page='help')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    name = request.form['name']
    email = request.form['email']
    comments = request.form['comments']
    db.add_feedback(name, email, comments)
    flash('Thank you for your feedback!', 'success')
    return redirect(url_for('help'))

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = db.authenticate_user(email, password)
        if user:
            session['user_id'] = user['id']
            session['email'] = email
            session['name'] = user['name']
            session['logged_in'] = True
            db.update_user_login(user['id'])
            flash(f'Welcome back, {user["name"] or email}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('index.html', page='signin')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        name = request.form.get('name', '')
        
        user_id = db.create_user(email, password, name)
        if user_id:
            session['user_id'] = user_id
            session['email'] = email
            session['name'] = name
            session['logged_in'] = True
            flash('Account created successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Email already exists. Please try logging in.', 'error')
    
    return render_template('index.html', page='signup')

@app.route('/signout')
def signout():
    session.clear()
    flash('You have been logged out successfully', 'info')
    return redirect(url_for('home'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'photo' in request.files:
            file = request.files['photo']
            if file and file.filename:
                filename = secure_filename(file.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)
                device_type = detect_device(path)
                questions = get_questions(device_type)
                return render_template('index.html', page='questions', questions=questions, device_type=device_type)
        # Fallback to manual if no photo
        device_type = request.form.get('device_type') or ''
        custom_device_type = request.form.get('custom_device_type') or ''
        final_device_type = custom_device_type.strip() if device_type == 'Other' and custom_device_type.strip() else device_type
        questions = get_questions(final_device_type)
        return render_template('index.html', page='questions', questions=questions, device_type=final_device_type)
    return render_template('index.html', page='upload')

@app.route('/recommendations', methods=['POST'])
@login_required
def recommendations():
    device_type = request.form['device_type']
    try:
        age = int(request.form['q0'])
    except ValueError:
        age = 0  # Default or error handle
    condition = request.form['q1']
    damage = request.form['q2']
    action = decide_action(device_type, age, condition, damage)
    rec = get_recommendations(action)
    eco_score = get_eco_score(action, device_type, age, condition)
    environmental_impact = get_environmental_impact(device_type, action)
    educational_tips = get_educational_tips(device_type)
    recycling_centers = db.get_recycling_centers() if action == 'Safe Disposal' else []
    
    # Store submission in database
    user_id = session['user_id']
    db.add_device_submission(
        user_id, device_type, age, condition, damage, 
        action, eco_score, environmental_impact
    )
    
    flash(f'Great! You earned {eco_score} eco points for {action.lower()} your {device_type.lower()}!', 'success')
    
    return render_template('index.html', 
                         page='recommendations', 
                         action=action, 
                         rec=rec, 
                         recycling_centers=recycling_centers,
                         eco_score=eco_score,
                         environmental_impact=environmental_impact,
                         educational_tips=educational_tips,
                         device_type=device_type)

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']
    user_submissions = db.get_user_submissions(user_id)
    user_stats = db.get_user_stats(user_id)
    
    # Calculate statistics
    device_counts = {}
    action_counts = {}
    for submission in user_submissions:
        device_counts[submission['device_type']] = device_counts.get(submission['device_type'], 0) + 1
        action_counts[submission['action_taken']] = action_counts.get(submission['action_taken'], 0) + 1
    
    return render_template('index.html', 
                         page='dashboard',
                         user_data=user_submissions,
                         total_eco_score=user_stats['total_eco_score'],
                         total_devices=user_stats['devices_processed'],
                         device_counts=device_counts,
                         action_counts=action_counts)

@app.route('/admin')
@admin_required
def admin_panel():
    feedbacks = db.get_all_feedback()
    return render_template('index.html', page='admin', feedbacks=feedbacks)

@app.route('/admin/feedback/<int:feedback_id>/<status>')
@admin_required
def update_feedback_status(feedback_id, status):
    db.update_feedback_status(feedback_id, status)
    flash(f'Feedback status updated to {status}', 'success')
    return redirect(url_for('admin_panel'))

@app.route('/api/recycling-centers')
def api_recycling_centers():
    """API endpoint for recycling centers (for map integration)"""
    centers = db.get_recycling_centers()
    return jsonify(centers)

@app.route('/map')
@login_required
def map_view():
    """Map view for recycling centers"""
    return render_template('index.html', page='map')

if __name__ == '__main__':
    app.run(debug=True)