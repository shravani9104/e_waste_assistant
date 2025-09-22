def decide_action(device_type, age, condition, damage):
    # age: int (years), condition: 'working'/'partial'/'broken', damage: 'yes'/'no'
    if condition == 'working':
        return 'Reuse/Donate'
    elif condition == 'partial' and damage == 'no':
        return 'Recycle'
    else:
        return 'Safe Disposal'

def get_questions(device_type):
    base_questions = [
        "What is the age of the device (in years)?",
        "What is the condition? (working/partial/broken)",
        "Is there any damage? (yes/no)"
    ]
    if device_type == 'battery':
        base_questions.append("Is it leaking? (yes/no)")
    return base_questions

def get_recommendations(action):
    recs = {
        'Reuse/Donate': 'Donate to charity or sell. Search for local donation centers.',
        'Recycle': 'Take to e-waste recycler. Remove batteries first.',
        'Safe Disposal': 'Use hazardous waste facility. Do not throw in trash.'
    }
    return recs.get(action, 'Unknown') + ' Eco-score: +10 points.'