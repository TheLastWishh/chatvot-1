import csv
import json

def get_disease_description(disease_name, file_path='E:/AI/Chatbot Medical v2/data/disease_description.csv'):
    """
    Lấy mô tả của một bệnh từ file CSV.
    
    Args:
        disease_name (str): Tên bệnh cần tìm
        file_path (str): Đường dẫn đến file CSV
        
    Returns:
        str: Mô tả của bệnh nếu tìm thấy, None nếu không tìm thấy
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[0].strip() == disease_name.strip():
                    return row[1]
        return None  # Trả về None nếu không tìm thấy bệnh
    except FileNotFoundError:
        print(f"Không tìm thấy file: {file_path}")
        return None
    except Exception as e:
        print(f"Lỗi khi đọc file: {str(e)}")
        return None

session_data = {
    'age': 21,
    'asked_symptoms': [],
    'current_symptoms': ['ngứa', 'chán_ăn', 'đôi_khi_buồn_nôn', 'da_vàng'],
    'gender': 'nam',
    'name': 'Nam',
    'state': 'END',
    'symptom': 'nước_tiểu_sẫm_màu',
    'prognosis': 'Ứ mật mãn tính'
}

def save_to_json(session_data):
    # Chuẩn bị dữ liệu theo định dạng yêu cầu
    user_data = {
        "Name": session_data.get('name', ''),
        "Age": session_data.get('age', 0),
        "Gender": "male" if session_data.get('gender', '').lower() == 'nam' else "female",
        "Disease": session_data.get('prognosis', ''),
        "Sympts": session_data.get('current_symptoms', [])
    }
    
    # Đọc file JSON hiện có (nếu tồn tại)
    try:
        with open('E:/AI/Chatbot Medical/DATA.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            # Kiểm tra xem 'users' có trong file JSON không
            if 'users' not in data:
                data['users'] = []
    except (FileNotFoundError, json.JSONDecodeError):
        # Nếu file không tồn tại hoặc rỗng, tạo mới
        data = {'users': []}
    
    # Thêm user mới vào danh sách
    data['users'].append(user_data)
    
    # Ghi lại vào file JSON
    with open('E:/AI/Chatbot Medical/DATA.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Ví dụ sử dụng:
# description = get_disease_description("AIDS")
# if description:
#     print(description)
# else:
#     print("Không tìm thấy mô tả cho bệnh này")