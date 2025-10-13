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

def get_recycling_centers():
    return [
        {
            'name': 'ERECON Recycling',
            'contact': '+91 9890863108 (S Shaikh)',
            'address': 'Head Office: Plot No. 53, Chikalthana, Jalna Road, Aurangabad, Maharashtra, India | Plant: Gut No. 94, Opp. R.L. Steel, Paithan Road, Chitegaon, Aurangabad, Maharashtra, India',
            'email': 'info@ereconrecycling.com',
            'website': 'www.ereconrecycling.com',
            'map': 'https://maps.google.com/?q=ERECON+Recycling+Aurangabad',
            'description': 'Professional e-waste and materials recycling company focusing on safe and efficient recycling practices.',
            'icon': '🏭'
        },
        {
            'name': 'The Kabadiwala',
            'contact': '098340 42258',
            'address': 'Bhakti Nagar, Harsul, Chhatrapati Sambhajinagar, Maharashtra 431008',
            'website': 'https://www.thekabadiwala.com/',
            'map': 'https://share.google/2nwX2q69zyYc1CKQq',
            'description': 'Digital platform for organized scrap collection and recycling services.',
            'icon': '♻️'
        },
        {
            'name': 'PERFECT E- WASTE RECYCLERS',
            'contact': '091566 06777',
            'address': 'Plot No. A, PERFECT E- WASTE RECYCLERS, 8/1, MIDC Industrial Area, Chilkalthana, Chhatrapati Sambhajinagar, Maharashtra 431006',
            'map': 'https://share.google/CmjAHToqKRV10PtUR',
            'description': 'Specialized e-waste recycling facility with professional disposal services.',
            'icon': '🔋'
        },
        {
            'name': 'Scrap Store',
            'contact': '08554930666',
            'address': 'Shop No.3, Plot no. 87, Sethi Complex, opp. Mahadev Mandir, Rokda Hanuman Colony, Rokdiya, Chhatrapati Sambhajinagar, Maharashtra 431001',
            'map': 'https://share.google/vGV19xKr0l2ylZvsq',
            'description': 'Organized scrap metal, plastic, paper, and electronics recycling center.',
            'icon': '🏪'
        },
        {
            'name': 'Semtronics Customer Care',
            'contact': '094222 22570',
            'address': 'Besides Atithi Hotel, Jyotirmaya complex, 23, Jalna Rd, Town Center, M G M, Chhatrapati Sambhajinagar, Maharashtra 431009',
            'map': 'https://share.google/hIDPijZmfuUBGm8mX',
            'description': 'Electronics service and recycling center with customer care support.',
            'icon': '📱'
        },
        {
            'name': 'Marathwada Scrap Centre',
            'contact': '098222 59601',
            'address': 'Kailash Nagar, Mondha, Chhatrapati Sambhajinagar, Maharashtra 431001',
            'map': 'https://share.google/7vUxpSD8yNxEIZxaa',
            'description': 'Local scrap collection and recycling center serving the Marathwada region.',
            'icon': '🏢'
        }
    ]

def get_recommendations(action):
    recs = {
        'Reuse/Donate': 'Donate to charity or sell. Search for local donation centers.',
        'Recycle': 'Take to e-waste recycler. Remove batteries first.',
        'Safe Disposal': 'Use hazardous waste facility. Do not throw in trash.'
    }
    return recs.get(action, 'Unknown') + ' Eco-score: +10 points.'

def get_eco_score(action, device_type, age, condition):
    """Calculate eco-score based on action and device details"""
    base_scores = {
        'Reuse/Donate': 15,
        'Recycle': 12,
        'Safe Disposal': 8
    }
    
    # Bonus points for newer devices
    age_bonus = max(0, 5 - age) if age <= 5 else 0
    
    # Bonus points for better condition
    condition_bonus = {
        'working': 5,
        'partial': 3,
        'broken': 0
    }.get(condition, 0)
    
    # Device type multiplier
    device_multiplier = {
        'battery': 1.5,  # Batteries are more hazardous
        'mobile': 1.2,
        'laptop': 1.3,
        'television': 1.4,
        'keyboard': 1.0,
        'player': 1.1
    }.get(device_type.lower(), 1.0)
    
    total_score = (base_scores.get(action, 5) + age_bonus + condition_bonus) * device_multiplier
    return int(total_score)

def get_environmental_impact(device_type, action):
    """Get environmental impact information"""
    impacts = {
        'battery': {
            'Reuse/Donate': 'Prevents 2.5kg CO2 emissions and saves 0.8kg of toxic materials from landfills',
            'Recycle': 'Recovers 95% of materials and prevents soil contamination',
            'Safe Disposal': 'Prevents 1.2kg of toxic materials from entering environment'
        },
        'mobile': {
            'Reuse/Donate': 'Extends device life by 2-3 years, saving 45kg CO2 emissions',
            'Recycle': 'Recovers 80% of precious metals and prevents e-waste pollution',
            'Safe Disposal': 'Prevents 0.5kg of toxic materials from contaminating soil'
        },
        'laptop': {
            'Reuse/Donate': 'Extends device life by 3-4 years, saving 120kg CO2 emissions',
            'Recycle': 'Recovers 85% of materials including rare earth metals',
            'Safe Disposal': 'Prevents 2kg of toxic materials from entering environment'
        },
        'television': {
            'Reuse/Donate': 'Extends device life by 4-5 years, saving 200kg CO2 emissions',
            'Recycle': 'Recovers 90% of materials including lead and mercury',
            'Safe Disposal': 'Prevents 5kg of toxic materials from contaminating environment'
        }
    }
    
    return impacts.get(device_type.lower(), {
        'Reuse/Donate': 'Extends device life and reduces manufacturing demand',
        'Recycle': 'Recovers valuable materials and prevents pollution',
        'Safe Disposal': 'Prevents toxic materials from entering environment'
    }).get(action, 'Helps protect the environment')

def get_educational_tips(device_type):
    """Get educational tips for specific device types"""
    tips = {
        'battery': [
            'Never throw batteries in regular trash - they contain toxic chemicals',
            'Store used batteries in a cool, dry place until disposal',
            'Consider rechargeable batteries to reduce waste',
            'Check for battery recycling programs in your area'
        ],
        'mobile': [
            'Backup your data before disposing of your phone',
            'Remove SIM card and memory card before recycling',
            'Factory reset to protect your personal information',
            'Consider donating to organizations that refurbish phones'
        ],
        'laptop': [
            'Wipe your hard drive completely before disposal',
            'Remove any personal stickers or identifying marks',
            'Check if components can be upgraded instead of replaced',
            'Consider donating to schools or non-profits'
        ],
        'television': [
            'Unplug and let it cool down before moving',
            'Remove any external devices and cables',
            'Check for manufacturer take-back programs',
            'Consider energy-efficient replacement options'
        ]
    }
    
    return tips.get(device_type.lower(), [
        'Always remove personal data before disposal',
        'Check for manufacturer recycling programs',
        'Consider donating if the device still works',
        'Research local recycling options in your area'
    ])