from flask import Flask, render_template, request, jsonify, send_file
import joblib
import pandas as pd
from src.utils import load_data
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
import os

app = Flask(__name__, template_folder='templates')

# Load model (same as before)
try:
    model_data = joblib.load('models/best_model.joblib')
    model = model_data['model']
    scaler = model_data['scaler']
    encoders = model_data.get('encoders', {})
except:
    model = None

df = load_data()
feature_cols = [col for col in df.columns if col != 'GradeClass']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded!'}), 500

    try:
        form_data = request.form.to_dict()
        
        input_dict = {}
        for col in feature_cols:
            value = form_data.get(col, '0')
            try:
                input_dict[col] = float(value)
            except:
                input_dict[col] = 0.0
        
        df_input = pd.DataFrame([input_dict])
        scaled = scaler.transform(df_input)
        
        pred = model.predict(scaled)[0]
        conf = model.predict_proba(scaled).max() * 100
        
        grade_map = {0: 'A (Excellent)', 1: 'B (Good)', 2: 'C (Average)', 3: 'D (Poor)', 4: 'F (Fail)'}
        predicted_grade = grade_map.get(pred, str(pred))
        
        # Generate PDF
        pdf_path = generate_pdf_report(form_data, predicted_grade, conf)
        
        return jsonify({
    'predicted_grade': predicted_grade,
    'confidence': round(conf, 1),
    'pdf_path': f"/download_report/{os.path.basename(pdf_path)}"
})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def generate_pdf_report(form_data, predicted_grade, confidence):
    os.makedirs('reports', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"Student_Performance_Report_{timestamp}.pdf"
    filepath = os.path.join('reports', filename)
    
    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Student Performance Prediction Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    y = height - 120
    c.drawString(50, y, "Student Input Data:")
    y -= 25
    for key, value in form_data.items():
        c.drawString(50, y, f"• {key}: {value}")
        y -= 20
    
    y -= 30
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"AI Prediction: {predicted_grade}")
    y -= 25
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Confidence Level: {confidence}%")
    c.drawString(50, y-30, "Note: This is an AI-generated prediction for academic guidance purposes.")
    
    c.save()
    return filepath

@app.route('/download_report/<path:filename>')
def download_report(filename):
    filepath = os.path.join('reports', filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True, download_name=filename)
    return "File not found", 404

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)