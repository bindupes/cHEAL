import os
import cv2
import numpy as np
import tensorflow as tf
from difflib import SequenceMatcher

# Global model variable
model = None

# Load model only when needed
def load_model():
    global model
    if model is None:
        model_path = r"C:\Users\17bin\OneDrive\Desktop\Documents\PERSONAL_GROWTH\cHEAL\17july\merge\iron_deficiency\final_dense_model.h5"
        model = tf.keras.models.load_model(model_path, compile=False)
    return model

# Classification function
def classify_image(image_path):
    model = load_model()
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array[..., ::-1]  # RGB -> BGR
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    pred = model.predict(img_array)
    label = "IDA" if pred[0][0] > 0.5 else "Normal"
    confidence = float(pred[0][0] if label == "IDA" else 1 - pred[0][0])
    return label, confidence

# Image feature extraction
def extract_ida_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > 50]
    area_thr = np.percentile(areas, 30) if areas else 0

    micro_count, hypo_count, total = 0, 0, len(areas)

    for c in cnts:
        A = cv2.contourArea(c)
        if A < 80:
            continue
        cell_mask = np.zeros_like(gray)
        cv2.drawContours(cell_mask, [c], -1, 255, -1)
        inner = cv2.erode(cell_mask, np.ones((7, 7), np.uint8), iterations=1)
        ring = cv2.subtract(cell_mask, inner)

        inner_mean = gray[inner > 0].mean() if inner.any() else 0
        ring_mean = gray[ring > 0].mean() if ring.any() else 0
        pallor_ratio = (inner_mean + 1) / (ring_mean + 1)

        if A <= area_thr:
            micro_count += 1
        if pallor_ratio >= 1.10:
            hypo_count += 1

    return {
        "total_cells": total,
        "microcytic_cells": micro_count,
        "hypochromic_cells": hypo_count,
        "abnormal_percent": round(((micro_count + hypo_count) / total * 100), 1) if total > 0 else 0
    }

# Spelling correction
def correct_spelling(user_q, keywords):
    words = user_q.lower().split()
    corrected, corrections = [], []
    for w in words:
        best = max(keywords, key=lambda k: SequenceMatcher(None, w, k).ratio(), default=w)
        if SequenceMatcher(None, w, best).ratio() > 0.8:
            corrected.append(best)
            if w != best:
                corrections.append((w, best))
        else:
            corrected.append(w)
    return " ".join(corrected), corrections

# Question understanding
def understand_question(user_q):
    keyword_map = {
        "definition": ["definition", "define", "meaning", "explain", "overview"],
        "symptoms": ["symptoms", "signs", "indications", "how do i know", "look for", "complaints"],
        "causes": ["causes", "why", "reason", "blood loss", "lead to", "etiology", "origin"],
        "tests": ["tests", "diagnosis", "investigations", "blood values"],
        "treatment": ["treatment", "cure", "management", "therapy", "dose", "iv iron", "medicines"],
        "diet": ["diet", "food", "nutrition", "eat", "avoid"],
        "differential": ["differential", "difference", "vs", "compare"],
        "followup": ["followup", "checkup", "monitor", "retest"],
        "pregnancy": ["pregnancy", "pregnant", "mother", "baby"],
        "children": ["children", "kids", "child", "infant", "growth", "pediatric"],
        "other_conditions": ["cold", "flu", "fever", "infection", "malaria", "cough", "virus", "illness"],
        "genetics": ["genetic", "hereditary", "family history", "inherited", "genes"],
        "lifestyle": ["lifestyle", "habits", "exercise", "rest", "daily routine", "activity", "sleep"],
        "emergencies": ["emergency", "urgent", "life threatening", "hospital", "danger", "critical", "severe"],
        "travel": ["travel", "flight", "journey", "trip", "altitude", "holiday", "safe to travel"],
        "risks": ["risks", "complications", "danger", "long term", "consequences", "problems"],
        "medicine":["medicine", "medicines", "drug", "tablet", "capsule", "pill"],
        "complications":["complications", "long term problems", "chronic issues", "side effects"],
        "prevention":["prevent", "avoid", "how to stop", "protection", "precaution"],
        "recovery":["how long", "recovery", "time", "duration", "heal", "improve"],
        "daily_life":["work", "job", "school", "exercise", "sports", "activities", "normal life"]

    }

    q_lower = user_q.lower()
    for q_type, synonyms in keyword_map.items():
        for phrase in synonyms:
            if phrase in q_lower:
                return q_type, q_lower, []

    return "definition", q_lower, []

# Response generator
def generate_response(q_type, label, conf, q, corrections):
    responses = {
        "definition": "📖 **Definition of IDA**\nIron-deficiency anemia (IDA) happens when there isn't enough iron to make hemoglobin.",
        "symptoms": "😷 **Common Symptoms of IDA**\n- Fatigue, pale skin, dizziness, brittle nails, pica",
        "causes": "🩸 **Causes of IDA**\n- Low iron intake, poor absorption, blood loss, increased demand",
        "tests": "🧪 **Tests for IDA**\n- CBC, Ferritin, Serum Iron, TIBC",
        "treatment": "💊 **Treatment of IDA**\n- Oral iron, IV iron if needed, treat underlying cause",
        "diet": "🍎 **Diet for IDA**\n- Iron-rich foods, Vitamin C, avoid tea/coffee with meals",
        "differential": "🔍 **IDA vs Thalassemia Trait**\n- Mentzer Index: >13 → IDA, <13 → Thalassemia trait\n- IDA: Low ferritin, responds to iron\n- Thalassemia trait: Normal ferritin, no response\n- Important for treatment decisions",
        "followup": "📆 **Follow-up in IDA**\n- Hb should rise ~1 g/dL every 2–3 weeks\n- Recheck Hb and ferritin after 2–3 months\n- Continue iron 3 months after Hb normalizes\n- Monitor recurrence if cause persists",
        "pregnancy": "🤰 **IDA in Pregnancy**\n- Risk of preterm delivery & low birth weight\n- Severe cases → maternal fatigue, cardiac stress\n- Prevention: iron + folic acid, regular Hb checks\n- Extra care in late pregnancy & postpartum",
        "children": "🧒 **IDA in Children**\n- Can cause poor growth, developmental delay\n- Symptoms: irritability, fatigue, poor concentration\n- Treatment: iron drops/syrup, diet improvement\n- Early treatment prevents long-term learning difficulties",
        "genetics": "🧬 **Genetics and IDA**\n- IDA not inherited; usually from diet or blood loss\n- Genetic conditions (thalassemia, celiac) can worsen anemia\n- Family history may increase risk indirectly",
        "lifestyle": "🏃 **Lifestyle & IDA**\n- Balanced diet with iron + vitamin C foods\n- Avoid skipping meals\n- Rest if fatigued\n- Moderate exercise once Hb improves\n- Avoid excess tea/coffee (reduces absorption)",
        "emergencies": "🚨 **When IDA Becomes an Emergency**\n- Very low Hb (<6–7 g/dL) → may need transfusion\n- Symptoms: chest pain, fainting, severe breathlessness\n- Hospital admission required",
        "travel": "✈️ **Travel with IDA**\n- Mild/moderate IDA → usually safe\n- Severe anemia → avoid flights (low oxygen risk)\n- Carry iron tablets, stay hydrated, avoid high altitudes if Hb very low",
        "risks": "⚠️ **Risks/Complications of Untreated IDA**\n- Severe fatigue, poor quality of life\n- Heart strain (palpitations, heart failure in chronic cases)\n- Pregnancy → low birth weight, preterm delivery\n- Children → growth & learning problems",
        "medicine": "💊 **Medicines for IDA**\n- First-line: Oral iron tablets (ferrous sulfate, gluconate, fumarate)\n- Take on empty stomach with Vitamin C for better absorption\n- Side effects: constipation, dark stools, nausea\n- Severe cases → IV iron injections or transfusion",
        "complications": "⚠️ **Complications of Untreated IDA**\n- Severe fatigue, poor concentration, poor quality of life\n- Heart strain → palpitations, heart failure (if chronic)\n- In children → poor growth, delayed learning\n- In pregnancy → miscarriage, low birth weight, maternal risk",
        "prevention": "🛡️ **Prevention of IDA**\n- Eat iron-rich diet (green leafy vegetables, meat, pulses)\n- Take Vitamin C with meals to improve absorption\n- Avoid excess tea/coffee with meals (reduces absorption)\n- Regular blood checkups if at risk (pregnant women, children, elderly)\n- Deworming in children where parasitic infections are common",
        "recovery": "⏳ **Recovery in IDA**\n- Hemoglobin usually rises by 1 g/dL every 2–3 weeks on treatment\n- Symptoms improve within 1–2 months\n- Full recovery of iron stores may take 3–6 months\n- Important: continue iron therapy for 3 months AFTER Hb normalizes to refill iron stores",
        "daily_life": "💼 **Daily Life with IDA**\n- Mild IDA → you can usually work, study, and exercise normally\n- Moderate/severe IDA → may need rest until Hb improves\n- Avoid heavy exercise until Hb is >10 g/dL\n- Children may feel tired → encourage balanced meals & regular checkups"


    }
    return responses.get(q_type, "🤔 I'm not sure. Try asking about symptoms, tests, treatment, or diet of IDA."), False

# VQA handler
def vqa_answer(question, image_path):
    try:
        label, conf = classify_image(image_path)
        q_type, corrected, corrections = understand_question(question)
        answer, _ = generate_response(q_type, label, conf, corrected, corrections)
        return answer, None
    except Exception as e:
        return f"Error processing question: {str(e)}", None

# Standalone testing
if __name__ == "__main__":
    image_path = input("Enter path to blood smear image: ")
    if not os.path.exists(image_path):
        print("❌ Image not found!")
        exit()

    # Classify
    label, conf = classify_image(image_path)
    print(f"✅ Detected: {label} ({conf*100:.1f}% confidence)")

    # Extract features
    features = extract_ida_features(image_path)
    print(f"📊 Features: {features['microcytic_cells']} microcytic, "
          f"{features['hypochromic_cells']} hypochromic "
          f"(~{features['abnormal_percent']}% abnormal cells)")

    # Ask questions
    while True:
        q = input("\nAsk a question about IDA (or type 'exit'): ").strip().lower()
        if q == "exit":
            break
        ans, _ = vqa_answer(q, image_path)
        print(ans)
