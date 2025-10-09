from flask import Flask, request, render_template, render_template_string, session, redirect, url_for
from decision_engine import decide_action, get_questions, get_recommendations
from model import detect_device
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change to a secure key
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

feedbacks = []  # Temporary storage; replace with database later

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
    feedbacks.append({'name': name, 'email': email, 'comments': comments})
    return redirect(url_for('help'))

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email == 'user@example.com' and password == 'password':
            session['logged_in'] = True
            return redirect(url_for('home'))
        return render_template('index.html', page='signin', error='Invalid credentials')
    return render_template('index.html', page='signin')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        session['logged_in'] = True
        return redirect(url_for('home'))
    return render_template('index.html', page='signup')

@app.route('/signout')
def signout():
    session.pop('logged_in', None)
    return redirect(url_for('home'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if not session.get('logged_in'):
        return redirect(url_for('signin'))
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
def recommendations():
    if not session.get('logged_in'):
        return redirect(url_for('signin'))
    device_type = request.form['device_type']
    try:
        age = int(request.form['q0'])
    except ValueError:
        age = 0  # Default or error handle
    condition = request.form['q1']
    damage = request.form['q2']
    action = decide_action(device_type, age, condition, damage)
    rec = get_recommendations(action)
    return render_template('index.html', page='recommendations', action=action, rec=rec)

if __name__ == '__main__':
    app.run(debug=True)