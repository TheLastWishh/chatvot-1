import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# if torch.cuda.is_available():
#     print(f"GPU khả dụng: {torch.cuda.get_device_name(0)}")
#     print(f"Số lượng GPU: {torch.cuda.device_count()}")
# else:
#     print("Không tìm thấy GPU hỗ trợ CUDA.")

def load_model_llm(model_path):
    # Tải tokenizer và mô hình
    print("Đang load mô hình...")
    # tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Đưa mô hình lên GPU nếu có
    if torch.cuda.is_available():
        model = model.cuda()
        
    model.eval()
    print("Load mô hình thành công!")
    return model, tokenizer

def extract_symptoms(text, model, tokenizer):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    
    # Chuyển inputs lên GPU nếu có
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Sinh kết quả
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )
    
    # Giải mã kết quả
    symptom_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Xử lý chuỗi trả về thành danh sách triệu chứng
    # Mẫu đầu ra dự kiến: "sốt cao, đau đầu" -> ["sốt cao", "đau đầu"]
    symptoms_list = [symptom.strip() for symptom in symptom_text.split(",")]
    
    # Loại bỏ các phần tử rỗng nếu có
    symptoms_list = list(set([symptom for symptom in symptoms_list if symptom]))
    
    return symptoms_list

# model_path = "E:/AI/Chatbot Medical v2/bartpho-syllable-base-final"
# model, tokenizer = load_model_llm(model_path)
# model, tokenizer = load_model_llm("Skirk383/bartpho-1")