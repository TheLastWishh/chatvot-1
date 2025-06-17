from flask import Flask, render_template, request, jsonify, session
import os
import sys
import secrets
import torch
import unicodedata

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.LLM import load_model_llm, extract_symptoms
from utils.DL import load_ann_model, suggest_symptoms_advanced, predict_disease
from utils.disease import get_disease_description, save_to_json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
llm_model, tokenizer = load_model_llm('Skirk383/bartpho-1')
ann_model = load_ann_model('model/disease_model.keras', 'model/encoders.pkl')

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

@app.route('/')
def home():
    session['current_symptoms'] = []
    session['related_symptoms'] = []
    session['symptom'] = 'none'
    session['state'] = 'free'
    session['name'] = 'none'
    session['age'] = 0
    session['gender'] = 'none'
    
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if request.method == 'POST':
        # Lấy dữ liệu từ request
        data = request.json
        print(data)
        msg = data.get('message', '')
        msg = unicodedata.normalize('NFC', msg)
        
        # Lấy trạng thái hiện tại từ session
        current_symptoms = session.get('current_symptoms', [])
        related_symptoms = session.get('related_symptoms', [])
        
        # Phản hồi mặc định
        response = 'Rất xin lỗi, tôi không hiểu bạn nói gì. Vui lòng tải lại cuộc trò chuyện!'
        
        if msg.upper() == "OK" and session['state'] == 'free':
            print('state: ', session['state'])
            response = "Vui lòng cho biết tên của bạn."
            session['state'] = 'ask_name'
            
        elif session['state'] == 'ask_name':
            print('state: ', session['state'])
            session['name'] = msg
            session['state'] = 'ask_age'
            response = 'Vui lòng cho biết tuổi của bạn.'
            
        elif session['state'] == 'ask_age':
            print('state: ', session['state'])
            if msg.isdigit():
                session['age'] = int(msg)
                session['state'] = 'ask_gender'
                response = 'Vui lòng cho biết giới tính của bạn.'
            else:
                response = 'Vui lòng nhập độ tuổi hợp lệ.'
        
        elif session['state'] == 'ask_gender':
            print('state: ', session['state'])
            if msg.lower() == unicodedata.normalize('NFC', 'nam') or msg.lower() == unicodedata.normalize('NFC', 'nữ'):
                session['gender'] = msg
                session['state'] = 'get_status_description' # get a description of your body condition.
                response = f"Xin chào {session['name']}, hãy cho biết tình trạng hiện tại của bạn!"
            else:
                response = 'Vui lòng nhập giới tính hợp lệ.'
        
        elif session['state'] == 'get_status_description':
            print('state: ', session['state'])
            print("Đang thực hiện trích xuất triệu chứng...")
            initial_symptoms = extract_symptoms(msg, model=llm_model, tokenizer=tokenizer)
            initial_symptoms = [unicodedata.normalize('NFC', symptom) for symptom in initial_symptoms]
            print("Trích xuất thành công: ", initial_symptoms)
            
            if initial_symptoms:
                # Cập nhật danh sách triệu chứng hiện tại
                current_symptoms.extend(initial_symptoms)
                
                # Lấy các triệu chứng liên quan
                related_symptoms = suggest_symptoms_advanced(initial_symptoms, ann_model, top_k=10)
                print(f'Danh sách các triệu chứng có thể gặp phải: {related_symptoms}')
                session['related_symptoms'] = related_symptoms
                session['state'] = 'ask_related_symptoms'
                session['symptom'] = related_symptoms[0]
                response = f"Bạn có gặp phải tình trạng {related_symptoms[0]} không?"
                print(f'Đặt câu hỏi về triệu chứng: {related_symptoms[0]}')
                del related_symptoms[0]
                
                
                # Lưu trạng thái mới vào session
                session['current_symptoms'] = current_symptoms
                session['related_symptoms'] = related_symptoms
            else:
                response = 'Rất xin lỗi, tôi không hiểu bạn nói gì. Vui lòng mô tả lại tình trạng của bạn!'
        
        elif session['state'] == 'ask_related_symptoms':
            if msg.lower() == 'yes' or msg.lower() == unicodedata.normalize('NFC', 'có'):
                current_symptoms.append(session['symptom'])
                print(f"session['symptom']: {session['symptom']}")
                print(f"related_symptoms: {related_symptoms}")
                print(f"current_symptoms: {current_symptoms}")
                
                if related_symptoms:
                    session['state'] == 'ask_related_symptoms'
                    session['symptom'] = related_symptoms[0]
                    response = f"Bạn có gặp phải tình trạng {related_symptoms[0]} không?"
                    print(f'Đặt câu hỏi về triệu chứng: {related_symptoms[0]}')
                    del related_symptoms[0]
                    
                else:
                    session['state'] = 'predict'
                    predict = predict_disease(current_symptoms)
                    disease = predict[0]['disease']
                    prob = predict[0]['probability']
                    
                    if prob >= 0.7:
                        print("State: ", session['state'])
                        print(f"disease: {disease}")
                        session['prognosis'] = disease
                        description = get_disease_description(disease)
                        response = f"Dựa trên thông tin bạn cung cấp, có thể bạn đang bị {disease}. <br> {description} <br> Lưu ý: Đây chỉ là chẩn đoán sơ bộ. Vui lòng tham khảo ý kiến của bác sĩ để được chẩn đoán và điều trị chính xác."
                        
                        # save_to_json(session)
                    else:
                        response = f"Rất xin lỗi, tôi không thể phán đoán được bệnh mà bạn đang gặp phải, vui lòng đến khám tại bệnh viện để có thể biết thêm về tình trạng của bản thân."
            
            elif msg.lower() == 'no' or msg.lower() == unicodedata.normalize('NFC', 'không'):
                print(f"session['symptom']: {session['symptom']}")
                print(f"related_symptoms: {related_symptoms}")
                print(f"current_symptoms: {current_symptoms}")
                
                if related_symptoms:
                    session['state'] == 'ask_related_symptoms'
                    session['symptom'] = related_symptoms[0]
                    response = f"Bạn có gặp phải tình trạng {related_symptoms[0]} không?"
                    print(f'Đặt câu hỏi về triệu chứng: {related_symptoms[0]}')
                    del related_symptoms[0]
                    
                else:
                    session['state'] = 'predict'
                    predict = predict_disease(current_symptoms)
                    disease = predict[0]['disease']
                    prob = predict[0]['probability']
                    
                    if prob >= 0.7:
                        print("State: ", session['state'])
                        print(f"disease: {disease}")
                        session['prognosis'] = disease
                        description = get_disease_description(disease)
                        response = f"Dựa trên thông tin bạn cung cấp, có thể bạn đang bị {disease}. \n\n {description} \n\n Lưu ý: Đây chỉ là chẩn đoán sơ bộ. Vui lòng tham khảo ý kiến của bác sĩ để được chẩn đoán và điều trị chính xác."
                        
                    else:
                        response = f"Rất xin lỗi, tôi không thể phán đoán được bệnh mà bạn đang gặp phải, vui lòng đến khám tại bệnh viện để có thể biết thêm về tình trạng của bản thân."
                        
            else:
                print(f"session['symptom']: {session['symptom']}")
                print(f"related_symptoms: {related_symptoms}")
                print(f"current_symptoms: {current_symptoms}")
                session['state'] == 'ask_related_symptoms'               
                symptom = session.get('symptom', 'none')
                response = f"Bạn có gặp phải tình trạng {symptom} không?"
        
        # Trả về phản hồi dưới dạng JSON
        return jsonify({
            'response': response
        })
        
@app.route('/reset', methods=['POST'])
def reset():
    """
        Reset cuộc trò chuyện
    """
    session['current_symptoms'] = []
    session['related_symptoms'] = []
    session['symptom'] = 'none'
    session['state'] = 'free'
    session['name'] = 'none'
    session['age'] = 0
    session['gender'] = 'none'
    
    return jsonify({'status': 'success', 'message': 'Đã reset cuộc trò chuyện'})


if __name__ == '__main__':
    app.run(debug=True)