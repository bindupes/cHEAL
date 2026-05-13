import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from difflib import SequenceMatcher
import webbrowser

# ================= MODEL LOADING =================
@st.cache_resource
def load_model():
    # model is in the same folder as this .py file
    model = tf.keras.models.load_model("final_dense_model.h5", compile=False)
    return model

model = load_model()


# ================= CLASSIFICATION =================
def classify_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))  # ✅ match model input
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array[..., ::-1]  # RGB -> BGR (like training)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    pred = model.predict(img_array)
    label = "IDA" if pred[0][0] > 0.5 else "Normal"
    confidence = float(pred[0][0] if label == "IDA" else 1 - pred[0][0])
    return label, confidence


# ================= IMAGE FEATURE EXTRACTION =================
def extract_ida_features(image_path):
    """Extract explainable features: counts of microcytic & hypochromic cells."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Preprocess
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > 50]
    area_thr = np.percentile(areas, 30) if areas else 0

    micro_count, hypo_count, total = 0, 0, len(areas)

    for c in cnts:
        A = cv2.contourArea(c)
        if A < 80:
            continue

        # create mask for central pallor
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


# ================= COLOR ANALYSIS HELPER =================
def open_color_tool():
    """Open local color analysis HTML tool."""
    tool_path = os.path.join(os.path.dirname(__file__), "color.html")  # relative path
    if os.path.exists(tool_path):
        webbrowser.open(f"file://{tool_path}")
    else:
        st.warning("⚠️ Color analysis tool not found. Please place 'color.html' in the same folder.")


# ================= SPELLING + QUESTION ROUTER =================
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



# ------------------- QUESTION ROUTER (with synonyms) -------------------
def understand_question(user_q):
    keyword_map = {
        "definition": ["definition", "what is", "define", "meaning", "explain", "overview"],
        "symptoms": ["symptoms", "signs", "indications", "how do i know", "look for", "complaints", "manifestations"],
        "causes": ["causes", "why", "reason", "blood loss", "lead to", "etiology", "origin"],
        "tests": ["tests", "diagnosis", "investigations", "blood values", "how do doctors confirm", "cbc", "ferritin", "lab results"],
        "treatment": ["treatment", "cure", "management", "therapy", "dose", "iv iron", "medicines", "how to treat"],
        "diet": ["diet", "food", "nutrition", "eat", "avoid", "good for", "dietary advice", "meal plan"],
        "differential": ["differential", "difference", "vs", "compare", "thalassemia", "mentzer", "look alike", "similar"],
        "followup": ["followup", "checkup", "monitor", "retest", "how often", "how long", "continue", "frequency"],
        "pregnancy": ["pregnancy", "pregnant", "mother", "baby", "dangerous in pregnancy", "prenatal", "antenatal"],
        "children": ["children", "kids", "child", "infant", "growth", "pediatric", "young"],
        "other_conditions": ["cold", "flu", "fever", "infection", "malaria", "cough", "virus", "illness"],
        "genetics": ["genetic", "hereditary", "family history", "inherited", "genes"],
        "lifestyle": ["lifestyle", "habits", "exercise", "rest", "daily routine", "activity", "sleep"],
        "emergencies": ["emergency", "urgent", "life threatening", "hospital", "danger", "critical", "severe"],
        "travel": ["travel", "flight", "journey", "trip", "altitude", "holiday", "safe to travel"],
        "risks": ["risks", "complications", "danger", "long term", "consequences", "problems"]
    }

    q_lower = user_q.lower()

    # Search for any keyword/synonym in the user question
    for q_type, synonyms in keyword_map.items():
        for phrase in synonyms:
            if phrase in q_lower:
                return q_type, q_lower, []

    # Default fallback
    return "definition", q_lower, []


# ================= ANSWER GENERATOR =================
def generate_response(q_type, label, conf, q, corrections):
    if q_type == "definition":
        return ("""
📖 **Definition of IDA**  
Iron-deficiency anemia (IDA) happens when there isn’t enough iron to make hemoglobin.  
- Red blood cells become **small (microcytic)**  
- They also appear **pale (hypochromic)** under the microscope.  
"""), False

    elif q_type == "symptoms":
        return ("""
😷 **Common Symptoms of IDA**  
- Fatigue and weakness  
- Pale skin (pallor)  
- Shortness of breath, dizziness, headaches  
- Brittle nails, hair loss  
- Pica (craving non-food items like ice or clay)  
"""), False

    elif q_type == "causes":
        return ("""
🩸 **Causes of IDA**  
- Low dietary intake of iron  
- Poor absorption (celiac disease, gastric surgery)  
- Chronic blood loss (GI ulcers, heavy periods)  
- Increased demand (pregnancy, growth in children)  
"""), False

    elif q_type == "tests":
        return ("""
🧪 **Tests for IDA**  
- CBC: ↓Hb, ↓MCV, ↓MCH, ↑RDW  
- Ferritin: Low  
- Serum Iron: Low  
- TIBC: High  
- Transferrin saturation: Low  
"""), False

    elif q_type == "treatment":
        return ("""
💊 **Treatment of IDA**  
- Oral iron supplements: 60–120 mg/day (adults)  
- Take with Vitamin C to improve absorption  
- Avoid tea, coffee, or calcium near the dose  
- IV iron if oral is not tolerated or ineffective  
- Always treat the underlying cause (e.g., bleeding)  
"""), False

    elif q_type == "diet":
        return ("""
🍎 **Diet for IDA**  
- Iron-rich foods: red meat, chicken, fish, legumes, leafy greens  
- Vitamin C foods (oranges, tomatoes) improve absorption  
- Avoid tea, coffee, excess calcium with iron meals  
- Enhancers: citrus fruits, fermented foods  
- Inhibitors: tea, coffee, phytates in grains  
"""), False

    elif q_type == "differential":
        return ("""
🔍 **IDA vs Thalassemia Trait**  
- Mentzer Index = MCV ÷ RBC count  
  • >13 → suggests IDA  
  • <13 → suggests Thalassemia trait  
- IDA: Low ferritin, responds to iron  
- Thalassemia trait: Normal ferritin, no response to iron  
- Important to distinguish → treatment approach differs  
"""), False

    elif q_type == "followup":
        return ("""
📆 **Follow-up in IDA**  
- Hb should rise ~1 g/dL every 2–3 weeks  
- Recheck Hb and ferritin after 2–3 months  
- Continue iron for 3 months after Hb normalizes to refill stores  
- Monitor for recurrence if underlying cause persists  
"""), False

    elif q_type == "pregnancy":
        return ("""
🤰 **IDA in Pregnancy**  
- Increases risk of preterm delivery & low birth weight  
- Severe cases → maternal fatigue, cardiac stress  
- Prevention: iron + folic acid supplementation, regular Hb checks  
- Special care needed during late pregnancy and postpartum  
"""), False

    elif q_type == "children":
        return ("""
🧒 **IDA in Children**  
- Can cause poor growth and developmental delay  
- Symptoms: irritability, fatigue, poor concentration  
- Treatment: iron drops/syrup, diet improvement  
- Early treatment prevents long-term learning difficulties  
"""), False

    elif q_type == "other_conditions":
        return ("""
⚠️ **IDA vs Other Conditions**  
- IDA = blood disorder due to low iron.  
- Cold/flu = infections caused by viruses.  
- They are unrelated. If symptoms overlap, a doctor can order blood tests.  
"""), False

    elif q_type == "genetics":
        return ("""
🧬 **Genetics and IDA**  
- IDA itself is not inherited → it usually develops from diet or blood loss.  
- But genetic conditions (like thalassemia or celiac disease) can worsen anemia.  
- Family history may increase risk indirectly.  
"""), False

    elif q_type == "lifestyle":
        return ("""
🏃 **Lifestyle & IDA**  
- Balanced diet with iron + vitamin C rich foods  
- Avoid skipping meals  
- Rest is important if fatigued  
- Moderate exercise is safe once Hb improves  
- Avoid excessive tea/coffee → reduces absorption  
"""), False

    elif q_type == "emergencies":
        return ("""
🚨 **When IDA Becomes an Emergency**  
- Very low Hb (<6–7 g/dL) → may need blood transfusion  
- Symptoms: chest pain, fainting, severe breathlessness  
- Hospital admission required in such cases  
"""), False

    elif q_type == "travel":
        return ("""
✈️ **Travel with IDA**  
- Mild/moderate IDA → usually safe to travel  
- Severe anemia → avoid flights until corrected (low oxygen risk)  
- Carry iron tablets, stay hydrated, avoid high altitude trips if Hb very low  
"""), False

    elif q_type == "risks":
        return ("""
⚠️ **Risks/Complications of Untreated IDA**  
- Severe fatigue and poor quality of life  
- Heart strain (palpitations, heart failure in chronic cases)  
- In pregnancy → low birth weight, preterm delivery  
- In children → growth and learning problems  
"""), False

    else:
        return ("🤔 I’m not sure. Try asking about symptoms, tests, treatment, or diet of IDA."), False


# ================= VQA HANDLER =================
def vqa_answer(question, image_path, label, conf):
    q_type, corrected, corrections = understand_question(question)
    answer, _ = generate_response(q_type, label, conf, corrected, corrections)
    return answer, None


# ================= STREAMLIT APP =================
st.title("Visual Question Answering (VQA) for Iron-Deficiency Anemia (IDA)")

uploaded = st.file_uploader("Upload a blood smear image", type=["jpg", "png", "jpeg"])
if uploaded:
    path = "temp.jpg"
    with open(path, "wb") as f:
        f.write(uploaded.read())

    # 1️⃣ Classify
    label, conf = classify_image(path)

    # 2️⃣ Extract explainable features
    features = extract_ida_features(path)

    # 3️⃣ Display results
    if label == "IDA":
        st.error(f"⚠️ IDA detected with {conf*100:.1f}% confidence")
        st.write(f"📊 Image features: {features['microcytic_cells']} microcytic, "
                 f"{features['hypochromic_cells']} hypochromic "
                 f"(~{features['abnormal_percent']}% abnormal cells)")
    else:
        st.success(f"✅ Normal with {conf*100:.1f}% confidence")
        st.write(f"📊 Image features: RBCs mostly normal size/pallor "
                 f"(abnormal ~{features['abnormal_percent']}%)")
        
    # 4️⃣ VQA interface
    q = st.text_input("Ask a question about IDA")
    if st.button("Submit") and q:
        ans, res_img = vqa_answer(q, path, label, conf)
        st.write(ans)

st.caption("⚠️ Educational demo only. Not for clinical diagnosis.")
