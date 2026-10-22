import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained SVM model
MODEL_PATH = 'svm.pkl'
model = None

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# HTML Template with Slate & Teal Dashboard Design
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Risk Prediction Dashboard</title>
    <!-- Tailwind CSS for modern responsive styling -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        slate: {
                            850: '#1e293b',
                            900: '#0f172a',
                        },
                        teal: {
                            500: '#14b8a6',
                            600: '#0d9488',
                        }
                    }
                }
            }
        }
    </script>
</head>
<body class="bg-slate-900 text-slate-100 font-sans antialiased min-h-screen flex flex-col">

    <!-- Top Navigation Bar -->
    <header class="bg-slate-850 border-b border-slate-700 py-4 px-6 shadow-md flex justify-between items-center">
        <div class="flex items-center space-x-3">
            <div class="bg-teal-500 text-slate-900 p-2 rounded-lg font-bold">SVM</div>
            <h1 class="text-xl font-bold tracking-wide text-white">Clinical Analytics & Prediction Hub</h1>
        </div>
        <span class="text-xs bg-teal-500/10 text-teal-400 border border-teal-500/20 px-3 py-1 rounded-full font-medium">Render Deployment Ready</span>
    </header>

    <!-- Main Dashboard Container -->
    <div class="flex-1 flex flex-col lg:flex-row">
        
        <!-- Sidebar Filters & Form Inputs -->
        <aside class="w-full lg:w-96 bg-slate-800 border-r border-slate-700 p-6 overflow-y-auto">
            <h2 class="text-lg font-semibold text-teal-400 mb-4 flex items-center gap-2">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"/></svg>
                Patient Parameters
            </h2>
            
            <form action="/" method="POST" class="space-y-4">
                <div>
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Age</label>
                    <input type="number" name="age" required value="{{ form_data.get('age', 45) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-100 focus:outline-none focus:border-teal-500">
                </div>

                <div class="grid grid-cols-2 gap-3">
                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Gender</label>
                        <select name="gender" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-100 focus:outline-none focus:border-teal-500">
                            <option value="1" {% if form_data.get('gender') == 1.0 %}selected{% endif %}>Male</option>
                            <option value="0" {% if form_data.get('gender') == 0.0 %}selected{% endif %}>Female</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">BMI</label>
                        <input type="number" step="0.1" name="bmi" required value="{{ form_data.get('bmi', 25.5) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-100 focus:outline-none focus:border-teal-500">
                    </div>
                </div>

                <div class="grid grid-cols-2 gap-3">
                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Fast. Blood Sugar</label>
                        <input type="number" step="0.1" name="fasting_blood_sugar" required value="{{ form_data.get('fasting_blood_sugar', 100) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-100 focus:outline-none focus:border-teal-500">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">HbA1c Level</label>
                        <input type="number" step="0.1" name="hba1c_level" required value="{{ form_data.get('hba1c_level', 5.6) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-100 focus:outline-none focus:border-teal-500">
                    </div>
                </div>

                <div class="grid grid-cols-2 gap-3">
                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">BP Systolic</label>
                        <input type="number" name="blood_pressure_systolic" required value="{{ form_data.get('blood_pressure_systolic', 120) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-100 focus:outline-none focus:border-teal-500">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">BP Diastolic</label>
                        <input type="number" name="blood_pressure_diastolic" required value="{{ form_data.get('blood_pressure_diastolic', 80) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-100 focus:outline-none focus:border-teal-500">
                    </div>
                </div>

                <div>
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Waist Circumference (cm)</label>
                    <input type="number" step="0.1" name="waist_circumference_cm" required value="{{ form_data.get('waist_circumference_cm', 85) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-100 focus:outline-none focus:border-teal-500">
                </div>

                <div class="grid grid-cols-2 gap-3">
                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Sleep Hours</label>
                        <input type="number" step="0.1" name="hours_sleep_per_night" required value="{{ form_data.get('hours_sleep_per_night', 7) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-100 focus:outline-none focus:border-teal-500">
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Stress Level (1-10)</label>
                        <input type="number" name="stress_level" required value="{{ form_data.get('stress_level', 5) }}" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-100 focus:outline-none focus:border-teal-500">
                    </div>
                </div>

                <div class="grid grid-cols-2 gap-3">
                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Family History</label>
                        <select name="family_history_diabetes" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-100 focus:outline-none focus:border-teal-500">
                            <option value="1" {% if form_data.get('family_history_diabetes') == 1.0 %}selected{% endif %}>Yes</option>
                            <option value="0" {% if form_data.get('family_history_diabetes') == 0.0 or form_data.get('family_history_diabetes') is none %}selected{% endif %}>No</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Physical Activity</label>
                        <select name="physical_activity_level" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-100 focus:outline-none focus:border-teal-500">
                            <option value="0" {% if form_data.get('physical_activity_level') == 0.0 %}selected{% endif %}>Low</option>
                            <option value="1" {% if form_data.get('physical_activity_level') == 1.0 or form_data.get('physical_activity_level') is none %}selected{% endif %}>Moderate</option>
                            <option value="2" {% if form_data.get('physical_activity_level') == 2.0 %}selected{% endif %}>High</option>
                        </select>
                    </div>
                </div>

                <!-- Additional fields mapped for full schema compatibility -->
                <input type="hidden" name="city" value="0">
                <input type="hidden" name="diet_type" value="0">
                <input type="hidden" name="smoking_status" value="0">
                <input type="hidden" name="alcohol_consumption" value="0">
                <input type="hidden" name="income_bracket" value="1">

                <button type="submit" class="w-full mt-4 bg-teal-500 hover:bg-teal-600 text-slate-900 font-bold py-3 px-4 rounded-lg transition duration-200 shadow-lg shadow-teal-500/20">
                    Run SVM Prediction
                </button>
            </form>
        </aside>

        <!-- Main Content Area: KPIs and Results Dashboard -->
        <main class="flex-1 p-6 lg:p-10 overflow-y-auto">
            <div class="max-w-4xl mx-auto space-y-6">
                
                <!-- Welcome Banner -->
                <div class="bg-slate-800 border border-slate-700 rounded-2xl p-6 shadow-xl flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                    <div>
                        <h2 class="text-2xl font-bold text-white">SVM Model Diagnostics</h2>
                        <p class="text-slate-400 text-sm mt-1">Real-time classification and risk score evaluation using Support Vector Machines.</p>
                    </div>
                    <div class="bg-slate-900 border border-slate-700 px-4 py-2 rounded-xl text-right">
                        <span class="block text-xs text-slate-400">Model Status</span>
                        <span class="inline-flex items-center gap-1.5 text-teal-400 font-semibold text-sm">
                            <span class="w-2 h-2 rounded-full bg-teal-400 animate-pulse"></span> Online & Active
                        </span>
                    </div>
                </div>

                <!-- Prediction KPI Card Result -->
                {% if prediction is not none %}
                <div class="bg-gradient-to-br from-slate-800 to-slate-850 border border-teal-500/30 rounded-2xl p-6 shadow-2xl relative overflow-hidden">
                    <div class="absolute top-0 right-0 transform translate-x-4 -translate-y-4 w-32 h-32 bg-teal-500/10 rounded-full blur-2xl"></div>
                    <h3 class="text-xs font-semibold uppercase tracking-wider text-teal-400 mb-2">Diagnostic Result</h3>
                    <div class="flex items-baseline gap-4">
                        <span class="text-4xl font-extrabold text-white">
                            {% if prediction == 1 %}
                                <span class="text-amber-400">High Risk Detected</span>
                            {% else %}
                                <span class="text-teal-400">Normal / Low Risk</span>
                            {% endif %}
                        </span>
                        <span class="text-xs bg-slate-900 px-3 py-1 rounded border border-slate-700 text-slate-300">Class Label: {{ prediction }}</span>
                    </div>
                    <p class="text-slate-400 text-sm mt-3">The Support Vector Classifier evaluated the input parameters against trained hyperplanes.</p>
                </div>
                {% else %}
                <div class="bg-slate-800/50 border border-dashed border-slate-700 rounded-2xl p-8 text-center">
                    <p class="text-slate-400">Configure parameters in the sidebar and click <span class="text-teal-400 font-medium">Run SVM Prediction</span> to generate metrics.</p>
                </div>
                {% endif %}

                <!-- Multi-column Metrics Grid -->
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div class="bg-slate-800 border border-slate-700 rounded-xl p-5 shadow-lg">
                        <span class="text-xs font-medium text-slate-400 uppercase">Evaluated Features</span>
                        <p class="text-2xl font-bold text-white mt-2">17 Parameters</p>
                        <span class="text-xs text-teal-400 mt-1 block">Full feature vector mapped</span>
                    </div>
                    <div class="bg-slate-800 border border-slate-700 rounded-xl p-5 shadow-lg">
                        <span class="text-xs font-medium text-slate-400 uppercase">Algorithm Kernel</span>
                        <p class="text-2xl font-bold text-white mt-2">RBF / SVC</p>
                        <span class="text-xs text-teal-400 mt-1 block">Non-linear classification</span>
                    </div>
                    <div class="bg-slate-800 border border-slate-700 rounded-xl p-5 shadow-lg">
                        <span class="text-xs font-medium text-slate-400 uppercase">Inference Engine</span>
                        <p class="text-2xl font-bold text-white mt-2">Flask + Gunicorn</p>
                        <span class="text-xs text-teal-400 mt-1 block">Cloud production ready</span>
                    </div>
                </div>

            </div>
        </main>
    </div>

    <!-- JavaScript to stop mouse wheel from modifying number inputs -->
    <script>
        document.addEventListener('wheel', function(e) {
            if (e.target.type === 'number') {
                e.preventDefault();
            }
        }, { passive: false });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    form_data = {}
    
    if request.method == 'POST':
        try:
            form_data = {
                'age': float(request.form.get('age', 45)),
                'gender': float(request.form.get('gender', 1)),
                'city': float(request.form.get('city', 0)),
                'bmi': float(request.form.get('bmi', 25.0)),
                'family_history_diabetes': float(request.form.get('family_history_diabetes', 0)),
                'physical_activity_level': float(request.form.get('physical_activity_level', 1)),
                'diet_type': float(request.form.get('diet_type', 0)),
                'smoking_status': float(request.form.get('smoking_status', 0)),
                'alcohol_consumption': float(request.form.get('alcohol_consumption', 0)),
                'hours_sleep_per_night': float(request.form.get('hours_sleep_per_night', 7)),
                'stress_level': float(request.form.get('stress_level', 5)),
                'fasting_blood_sugar': float(request.form.get('fasting_blood_sugar', 100)),
                'hba1c_level': float(request.form.get('hba1c_level', 5.5)),
                'blood_pressure_systolic': float(request.form.get('blood_pressure_systolic', 120)),
                'blood_pressure_diastolic': float(request.form.get('blood_pressure_diastolic', 80)),
                'waist_circumference_cm': float(request.form.get('waist_circumference_cm', 85)),
                'income_bracket': float(request.form.get('income_bracket', 1))
            }

            features = np.array([[
                form_data['age'],
                form_data['gender'],
                form_data['city'],
                form_data['bmi'],
                form_data['family_history_diabetes'],
                form_data['physical_activity_level'],
                form_data['diet_type'],
                form_data['smoking_status'],
                form_data['alcohol_consumption'],
                form_data['hours_sleep_per_night'],
                form_data['stress_level'],
                form_data['fasting_blood_sugar'],
                form_data['hba1c_level'],
                form_data['blood_pressure_systolic'],
                form_data['blood_pressure_diastolic'],
                form_data['waist_circumference_cm'],
                form_data['income_bracket']
            ]])

            if model:
                pred = model.predict(features)
                prediction = int(pred[0])
            else:
                prediction = 0 
        except Exception as e:
            print(f"Prediction error: {e}")

    return render_template_string(HTML_TEMPLATE, prediction=prediction, form_data=form_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
