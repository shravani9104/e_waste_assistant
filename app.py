from flask import Flask, request, render_template_string
from model import detect_device
from decision_engine import decide_action, get_questions, get_recommendations
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'photo' in request.files:
            file = request.files['photo']
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            device_type = detect_device(path)
            questions = get_questions(device_type)
            return render_template_string(QUESTIONS_TEMPLATE, questions=questions, device_type=device_type)

        elif 'answers' in request.form:
            device_type = request.form['device_type']
            age = int(request.form['q0'])
            condition = request.form['q1']
            damage = request.form['q2']
            action = decide_action(device_type, age, condition, damage)
            rec = get_recommendations(action)
            return render_template_string(REC_TEMPLATE, action=action, rec=rec)

    return render_template_string(UPLOAD_TEMPLATE)

UPLOAD_TEMPLATE = '''
<form method="post" enctype="multipart/form-data">
    Upload photo: <input type="file" name="photo">
    <input type="submit">
</form>
'''

QUESTIONS_TEMPLATE = '''
Detected: {{ device_type }}
<form method="post">
    {% for q in questions %}
        {{ q }} <input name="q{{ loop.index0 }}"><br>
    {% endfor %}
    <input type="hidden" name="device_type" value="{{ device_type }}">
    <input type="hidden" name="answers" value="1">
    <input type="submit">
</form>
'''

REC_TEMPLATE = '''
Action: {{ action }}<br>
Recommendations: {{ rec }}
'''

if __name__ == '__main__':
    app.run(debug=True)