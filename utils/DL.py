import pickle
import numpy as np
from collections import defaultdict
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def load_ann_model(model_path='E:/ĐATN/Chatbot Medical/model/disease_model.keras', encoders_path='E:/AI/Chatbot Medical v2/model/encoders.pkl'):
    global MODEL, SYMPTOM_ENCODER, DISEASE_ENCODER
    global SYMPTOM_TO_IDX, IDX_TO_SYMPTOM, DISEASE_TO_IDX, IDX_TO_DISEASE
    global SYMPTOM_COOCCURRENCE, SYMPTOM_EMBEDDINGS, SYMPTOM_SIMILARITY_MATRIX, SYMPTOM_NAMES, DISEASE_SYMPTOM_MAP
    MODEL = tf.keras.models.load_model(model_path)
    with open(encoders_path, 'rb') as f:
        data = pickle.load(f)
        SYMPTOM_ENCODER = data['symptom_encoder']
        DISEASE_ENCODER = data['disease_encoder']
        SYMPTOM_TO_IDX = data['symptom_to_idx']
        IDX_TO_SYMPTOM = data['idx_to_symptom']
        DISEASE_TO_IDX = data['disease_to_idx']
        IDX_TO_DISEASE = data['idx_to_disease']
        SYMPTOM_COOCCURRENCE = data['symptom_cooccurrence']
        SYMPTOM_EMBEDDINGS = data['symptom_embeddings']
        SYMPTOM_SIMILARITY_MATRIX = data['symptom_similarity_matrix']
        SYMPTOM_NAMES = data['symptoms_names']
        DISEASE_SYMPTOM_MAP = data['disease_symptom_map']

    return MODEL
# ===== HÀM DỰ ĐOÁN BỆNH =====
def predict_disease(symptoms):
    """Dự đoán bệnh từ các triệu chứng nhập vào"""
    if isinstance(symptoms, str):
        symptoms = [symptoms]
    
    # Kiểm tra và cảnh báo triệu chứng không có trong encoder
    valid_symptoms = []
    for symptom in symptoms:
        if symptom in SYMPTOM_ENCODER.classes_:
            valid_symptoms.append(symptom)
        else:
            print(f"Cảnh báo: Triệu chứng '{symptom}' không có trong danh sách đã biết.")
    
    # Nếu không có triệu chứng hợp lệ nào
    if not valid_symptoms:
        print("Lỗi: Không có triệu chứng hợp lệ nào để thực hiện dự đoán!")
        return []
    
    # Mã hóa triệu chứng
    symptom_vector = SYMPTOM_ENCODER.transform([valid_symptoms])
    
    # Dự đoán
    probabilities = MODEL.predict(symptom_vector, verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:10]

    return [{
        'disease': DISEASE_ENCODER.classes_[i],
        'probability': float(probabilities[i])
    } for i in top_indices] 
        
# ===== 1. CHUẨN HÓA ĐIỂM SỐ =====
def normalize_scores(scores_dict, method='minmax'):
    """
    Chuẩn hóa điểm số về khoảng [0, 1]
    
    Args:
        scores_dict: Dictionary {triệu_chứng: điểm_số}
        method: 'minmax' hoặc 'zscore'
    
    Returns:
        Dictionary với điểm số đã chuẩn hóa
    """
    if not scores_dict:
        return {}
    
    scores = np.array(list(scores_dict.values())).reshape(-1, 1)
    symptoms = list(scores_dict.keys())
    
    if method == 'minmax':
        scaler = MinMaxScaler()
        normalized_scores = scaler.fit_transform(scores).flatten()
    elif method == 'zscore':
        normalized_scores = (scores.flatten() - np.mean(scores)) / (np.std(scores) + 1e-8)
        # Chuyển về [0, 1] bằng sigmoid
        normalized_scores = 1 / (1 + np.exp(-normalized_scores))
    
    return dict(zip(symptoms, normalized_scores))

# ===== 2. TẠO ĐIỂM SỐ CHO PHƯƠNG PHÁP DỰ ĐOÁN BỆNH =====
def create_disease_based_scores(input_symptoms, model, method='probability_weighted'):
    """
    Tạo điểm số cho triệu chứng dựa trên dự đoán bệnh
    
    Args:
        input_symptoms: List triệu chứng đầu vào
        model: Mô hình đã huấn luyện
        method: Phương pháp tính điểm
            - 'probability_weighted': Điểm = xác suất bệnh * trọng số triệu chứng
            - 'frequency_based': Dựa trên tần suất xuất hiện trong bệnh
            - 'rank_based': Dựa trên thứ hạng trong top diseases
    
    Returns:
        Dictionary {triệu_chứng: điểm_số}
    """
    # Mã hóa triệu chứng đầu vào
    input_encoded = SYMPTOM_ENCODER.transform([input_symptoms])
    
    # Dự đoán bệnh
    disease_probs = model.predict(input_encoded, verbose=0)[0]
    top_diseases_idx = np.argsort(disease_probs)[-5:][::-1]  # Top 5 bệnh
    
    symptom_scores = defaultdict(float)
    
    for rank, disease_idx in enumerate(top_diseases_idx):
        disease_name = IDX_TO_DISEASE[disease_idx]
        disease_prob = disease_probs[disease_idx]
        
        if disease_name in DISEASE_SYMPTOM_MAP:
            disease_symptoms = DISEASE_SYMPTOM_MAP[disease_name]
            
            for symptom in disease_symptoms:
                if symptom not in input_symptoms:
                    if method == 'probability_weighted':
                        # Điểm = xác suất bệnh * trọng số vị trí (cao hơn cho bệnh có xác suất cao)
                        weight = disease_prob * (1.0 - rank * 0.1)  # Giảm trọng số theo rank
                        symptom_scores[symptom] += weight
                        
                    elif method == 'frequency_based':
                        # Điểm dựa trên tần suất xuất hiện
                        symptom_scores[symptom] += 1.0 * disease_prob
                        
                    elif method == 'rank_based':
                        # Điểm dựa trên thứ hạng (rank 0 = điểm cao nhất)
                        rank_score = 1.0 / (rank + 1)
                        symptom_scores[symptom] += rank_score * disease_prob
    
    return dict(symptom_scores)

# ===== 3. PHƯƠNG PHÁP ĐỒNG XUẤT HIỆN CẢI TIẾN =====
def get_cooccurrence_scores(input_symptoms, method='tfidf_weighted'):
    """
    Tính điểm đồng xuất hiện với các cải tiến
    
    Args:
        method: 'simple', 'tfidf_weighted', 'jaccard'
    """
    related_scores = defaultdict(float)
    
    for symptom in input_symptoms:
        if symptom in SYMPTOM_COOCCURRENCE:
            for related_symptom, co_count in SYMPTOM_COOCCURRENCE[symptom].items():
                if related_symptom not in input_symptoms:
                    
                    if method == 'simple':
                        related_scores[related_symptom] += co_count
                        
                    elif method == 'tfidf_weighted':
                        # Áp dụng TF-IDF weighting
                        tf = co_count
                        # IDF: log(tổng số triệu chứng / số triệu chứng chứa related_symptom)
                        df = sum(1 for s in SYMPTOM_COOCCURRENCE.values() 
                                if related_symptom in s)
                        idf = np.log(len(SYMPTOM_COOCCURRENCE) / (df + 1))
                        tfidf_score = tf * idf
                        related_scores[related_symptom] += tfidf_score
                        
                    elif method == 'jaccard':
                        # Jaccard similarity
                        intersection = co_count
                        union = (len(SYMPTOM_COOCCURRENCE[symptom]) + 
                                len(SYMPTOM_COOCCURRENCE.get(related_symptom, {})) - 
                                intersection)
                        jaccard_score = intersection / (union + 1e-8)
                        related_scores[related_symptom] += jaccard_score
    
    return dict(related_scores)

# ===== 4. PHƯƠNG PHÁP EMBEDDING CẢI TIẾN =====
def get_embedding_scores(input_symptoms, method='weighted_average'):
    """
    Tính điểm similarity từ embedding với cải tiến
    
    Args:
        method: 'weighted_average', 'max_similarity', 'attention_weighted'
    """
    if SYMPTOM_SIMILARITY_MATRIX is None:
        return {}
    
    related_scores = defaultdict(float)
    input_indices = []
    
    # Tìm indices của input symptoms
    for symptom in input_symptoms:
        if symptom in SYMPTOM_NAMES:
            input_indices.append(SYMPTOM_NAMES.index(symptom))
    
    if not input_indices:
        return {}
    
    for i, symptom_name in enumerate(SYMPTOM_NAMES):
        if symptom_name not in input_symptoms:
            similarities = []
            
            for input_idx in input_indices:
                sim = SYMPTOM_SIMILARITY_MATRIX[input_idx][i]
                similarities.append(sim)
            
            if method == 'weighted_average':
                # Trung bình có trọng số (trọng số cao hơn cho similarity cao)
                weights = np.array(similarities)
                weights = weights / (np.sum(weights) + 1e-8)
                score = np.sum(weights * similarities)
                
            elif method == 'max_similarity':
                # Lấy similarity cao nhất
                score = max(similarities)
                
            elif method == 'attention_weighted':
                # Attention mechanism đơn giản
                attention_weights = np.exp(similarities) / (np.sum(np.exp(similarities)) + 1e-8)
                score = np.sum(attention_weights * similarities)
            
            if score > 0.1:  # Threshold để lọc noise
                related_scores[symptom_name] = score
    
    return dict(related_scores)

# ===== 5. ENSEMBLE METHODS =====
def ensemble_symptom_suggestions(input_symptoms, model, 
                                weights=[0.4, 0.3, 0.3], 
                                top_k=10,
                                diversity_penalty=0.1):
    """
    Kết hợp 3 phương pháp với ensemble learning
    
    Args:
        weights: [weight_cooccurrence, weight_embedding, weight_disease]
        diversity_penalty: Hệ số penalty cho các triệu chứng xuất hiện ở nhiều phương pháp
    """
    
    # 1. Lấy điểm từ 3 phương pháp
    cooccurrence_scores = get_cooccurrence_scores(input_symptoms, method='tfidf_weighted')
    embedding_scores = get_embedding_scores(input_symptoms, method='attention_weighted')
    disease_scores = create_disease_based_scores(input_symptoms, model, method='probability_weighted')
    
    # 2. Chuẩn hóa điểm số
    cooccurrence_norm = normalize_scores(cooccurrence_scores, method='minmax')
    embedding_norm = normalize_scores(embedding_scores, method='minmax')
    disease_norm = normalize_scores(disease_scores, method='minmax')
    
    # 3. Kết hợp các triệu chứng
    all_symptoms = set()
    all_symptoms.update(cooccurrence_norm.keys())
    all_symptoms.update(embedding_norm.keys())
    all_symptoms.update(disease_norm.keys())
    
    # 4. Tính điểm tổng hợp
    final_scores = {}
    method_counts = defaultdict(int)  # Đếm số phương pháp gợi ý mỗi triệu chứng
    
    for symptom in all_symptoms:
        score = 0.0
        count = 0
        
        if symptom in cooccurrence_norm:
            score += weights[0] * cooccurrence_norm[symptom]
            count += 1
            
        if symptom in embedding_norm:
            score += weights[1] * embedding_norm[symptom]
            count += 1
            
        if symptom in disease_norm:
            score += weights[2] * disease_norm[symptom]
            count += 1
        
        method_counts[symptom] = count
        
        # Áp dụng diversity bonus/penalty
        if count > 1:
            # Bonus cho triệu chứng được nhiều phương pháp gợi ý
            diversity_bonus = (count - 1) * diversity_penalty
            score += diversity_bonus
        
        final_scores[symptom] = score
    
    # 5. Sắp xếp và trả về top-k
    sorted_symptoms = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 6. Tạo kết quả chi tiết
    results = []
    for symptom, final_score in sorted_symptoms[:top_k]:
        detail = {
            'symptom': symptom,
            'final_score': final_score,
            'method_count': method_counts[symptom],
            'cooccurrence_score': cooccurrence_norm.get(symptom, 0.0),
            'embedding_score': embedding_norm.get(symptom, 0.0),
            'disease_score': disease_norm.get(symptom, 0.0)
        }
        results.append(detail)
    
    return results

# ===== 6. ADAPTIVE WEIGHTS =====
def adaptive_ensemble_weights(input_symptoms, model):
    """
    Tự động điều chỉnh trọng số dựa trên đặc điểm của input
    """
    base_weights = [0.4, 0.3, 0.3]  # [cooccurrence, embedding, disease]
    
    # Điều chỉnh dựa trên số lượng triệu chứng đầu vào
    num_symptoms = len(input_symptoms)
    
    if num_symptoms == 1:
        # Với 1 triệu chứng, ưu tiên cooccurrence và disease prediction
        weights = [0.5, 0.2, 0.3]
    elif num_symptoms <= 3:
        # Với ít triệu chứng, cân bằng 3 phương pháp
        weights = [0.4, 0.3, 0.3]
    else:
        # Với nhiều triệu chứng, ưu tiên embedding và disease prediction
        weights = [0.3, 0.4, 0.3]
    
    # Điều chỉnh dựa trên confidence của disease prediction
    input_encoded = SYMPTOM_ENCODER.transform([input_symptoms])
    disease_probs = model.predict(input_encoded, verbose=0)[0]
    max_disease_prob = np.max(disease_probs)
    
    if max_disease_prob > 0.8:
        # Nếu dự đoán bệnh rất tự tin, tăng trọng số cho disease-based
        weights[2] += 0.1
        weights[0] -= 0.05
        weights[1] -= 0.05
    elif max_disease_prob < 0.3:
        # Nếu dự đoán bệnh không tự tin, giảm trọng số cho disease-based
        weights[2] -= 0.1
        weights[0] += 0.05
        weights[1] += 0.05
    
    # Đảm bảo tổng trọng số = 1
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    return weights.tolist()

# ===== 7. HÀM CHÍNH =====
def suggest_symptoms_advanced(input_symptoms, model, top_k=10, use_adaptive_weights=True):
    """
    Hàm chính để gợi ý triệu chứng với các tính năng nâng cao
    """
    print(f"Đang phân tích triệu chứng đầu vào: {input_symptoms}")
    
    # Xác định trọng số
    if use_adaptive_weights:
        weights = adaptive_ensemble_weights(input_symptoms, model)
        print(f"Trọng số tự động: Cooccurrence={weights[0]:.2f}, "
              f"Embedding={weights[1]:.2f}, Disease={weights[2]:.2f}")
    else:
        weights = [0.4, 0.3, 0.3]
    
    # Thực hiện ensemble
    results = ensemble_symptom_suggestions(
        input_symptoms, model, 
        weights=weights, 
        top_k=top_k,
        diversity_penalty=0.1
    )
    
    # Lấy danh sách triệu chứng từ results
    symptom_list = [result['symptom'] for result in results]
    
    return symptom_list

# # ===== 8. HÀM HIỂN THỊ KẾT QUẢ =====
# def display_suggestions(results):
#     """Hiển thị kết quả gợi ý một cách đẹp mắt"""
#     print("\n" + "="*80)
#     print("                    KẾT QUẢ GỢI Ý TRIỆU CHỨNG")
#     print("="*80)
    
#     for i, result in enumerate(results, 1):
#         print(f"\n{i:2d}. {result['symptom']}")
#         print(f"    Điểm tổng hợp: {result['final_score']:.4f}")
#         print(f"    Số phương pháp gợi ý: {result['method_count']}/3")
#         print(f"    Chi tiết điểm:")
#         print(f"      - Đồng xuất hiện: {result['cooccurrence_score']:.3f}")
#         print(f"      - Embedding:      {result['embedding_score']:.3f}")
#         print(f"      - Dự đoán bệnh:   {result['disease_score']:.3f}")

# model = load_ann_model()        
# input_symptoms = ["ho khan", "sốt nhẹ"]
# results = suggest_symptoms_advanced(input_symptoms, model, top_k=10)
# print(results)

# display_suggestions(results)