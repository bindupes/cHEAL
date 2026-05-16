import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import re
from difflib import SequenceMatcher
import webbrowser
import subprocess
import threading
import time

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("D:/model_fold5.h5")

model = load_model()

# Prediction function
def classify_image(image):
    # Convert to numpy (PIL -> array) and ensure same color order
    img = np.array(image)

    # If your training used cv2 (BGR), convert RGB->BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize exactly as in training
    img = cv2.resize(img, (224, 224))

    # Normalize (make sure same as training)
    img = img / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img, axis=0)

    prediction = model.predict(img_array)

    # Interpret sigmoid output
    label = "Sickle Cell" if prediction[0][0] > 0.5 else "Normal Cell"
    confidence = round(float(prediction[0][0] if label == "Sickle Cell" else 1 - prediction[0][0]), 2)

    return label, confidence


# Highlight sickle cells only
def highlight_cells(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    h, w = binary.shape
    center_x, center_y = w // 2, h // 2
    radius = min(h, w) // 2 - 30
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    masked_binary = cv2.bitwise_and(binary, binary, mask=mask)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(masked_binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 300 < area < 8000 and len(cnt) >= 5:
            try:
                (x_, y_), (MA, ma), angle = cv2.fitEllipse(cnt)
                eccentricity = np.sqrt(1 - (min(MA, ma) / max(MA, ma))**2)
            except:
                continue

            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(cnt)

            corners = [
                (x, y), (x + w, y), (x, y + h), (x + w, y + h)
            ]

            all_corners_inside = all(
                np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2) <= radius for cx, cy in corners
            )

            if all_corners_inside:
                if eccentricity > 0.88 and len(approx) < 7:
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 3)
                elif 0.82 < eccentricity <= 0.88 and len(approx) < 9:
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 165, 255), 3)

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output_rgb

def open_color_analysis_tool():
    """Opens the color analysis HTML tool in default browser"""
    try:
        # Get the absolute path to color.html
        color_html_path = os.path.abspath("color.html")
        
        if os.path.exists(color_html_path):
            # Open in default browser
            webbrowser.open(r"C:\Users\17bin\OneDrive\Desktop\Documents\PERSONAL_GROWTH\cHEAL\17july\color.html")
            return True
        else:
            st.error("❌ color.html file not found! Please make sure it's in the same directory as this script.")
            return False
    except Exception as e:
        st.error(f"❌ Error opening color analysis tool: {str(e)}")
        return False
    
def create_color_analysis_button():
    """Creates a button that opens the color analysis tool"""
    st.markdown("### 🎨 Advanced Color Analysis Tool")
    st.info("For detailed RBC color analysis and stain identification, use our specialized tool:")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔬 Open Color Analysis Tool", key="color_tool_btn", help="Opens color.html in your browser"):
            with st.spinner("Opening color analysis tool..."):
                if open_color_analysis_tool():
                    st.success("✅ Color analysis tool opened in your browser!")
                    st.markdown("""
                    **Instructions:**
                    1. Upload your blood smear image in the opened tool
                    2. Click 'Analyze RBC Colors' 
                    3. Get detailed color analysis and stain identification
                    4. Download the analysis report if needed
                    """)
                else:
                    st.error("Failed to open the color analysis tool.")


# Spell check function
def check_spelling_and_suggest(text, common_words):
    words = text.lower().split()
    corrected_words = []
    corrections_made = []
    
    for word in words:
        best_match = None
        best_ratio = 0.85  # Higher threshold for similarity to avoid false corrections
        
        # Skip correction if word is too short or looks correct
        if len(word) < 4:
            corrected_words.append(word)
            continue
            
        for common_word in common_words:
            ratio = SequenceMatcher(None, word, common_word).ratio()
            if ratio > best_ratio and len(common_word) > 3:
                best_match = common_word
                best_ratio = ratio
        
        if best_match and word != best_match and best_ratio > 0.85:
            corrected_words.append(best_match)
            corrections_made.append(f"'{word}' → '{best_match}'")
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words), corrections_made

# Enhanced Natural language understanding for questions
def understand_question(question):
    question = question.lower().strip()
    
    # Expanded common medical terms for spell checking
    common_words = [
        'highlight', 'show', 'sickle', 'cell', 'cells', 'symptoms', 'anemia', 'anaemia',
        'treatment', 'genetic', 'dangerous', 'shape', 'normal', 'abnormal', 'infection',
        'infected', 'stages', 'food', 'diet', 'causes', 'patient', 'healthy', 'disease',
        'harmful', 'permanent', 'cure', 'lifespan', 'common', 'activities', 'contagious',
        'sexual', 'partner', 'baby', 'inherit', 'age', 'treatment', 'bone', 'marrow',
        'difference', 'trait', 'boys', 'girls', 'early', 'untreated', 'pain', 'crisis',
        'organs', 'growth', 'development', 'fertility', 'birth', 'risks', 'lifestyle',
        'country', 'types', 'blood', 'smear', 'checkups', 'tests', 'ayurvedic', 'travel',
        'doctor', 'specialist', 'hospital', 'home', 'remedy', 'breastfeed', 'nursing', 'mental', 'depression', 'anxiety', 'coping', 'emotional', 'support', 'insurance',
        'cost', 'financial', 'education', 'school', 'work', 'employment', 'emergency',
        'therapy', 'alternative', 'adult', 'clinical', 'trial', 'aging', 'health','elderly'
    ]
    
    corrected_question, corrections_made = check_spelling_and_suggest(question, common_words)
    
    color_keywords = [
        'what is color of rbc', 'what is color of cell', 'what is the color of rbc',
        'what is the color of cell', 'color of rbc', 'color of cell', 'color of cells',
        'what color are the rbc', 'what color are the cells', 'rbc color', 'cell color',
        'what stain', 'which stain', 'stain used', 'staining', 'identify stain',
        'color analysis', 'analyse color', 'analyze color'
    ]
    
    if any(phrase in corrected_question for phrase in color_keywords):
        return 'color_analysis', corrected_question, corrections_made
    

    # NEW QUESTION TYPES - Priority order matters!
    
    # Image content question
    if any(phrase in corrected_question for phrase in ['what is it in the image', 'what do you see in the image', 'describe the image', 'what is in this image']):
        return 'image_content', corrected_question, corrections_made
    
    if any(phrase in corrected_question for phrase in ['what causes sickle cell', 'cause of sickle cell', 'why sickle cell happens', 'why does sickle cell occur', 'what causes sickle cell disease']):
        return 'causes', corrected_question, corrections_made

    # NEW: Definition / difference between terms
    if any(phrase in corrected_question for phrase in ['what is sickle cell', 'what is sickle cell disease', 'what is sickle cell anaemia', 'what is sickle cell anemia', 'define sickle cell', 'explain sickle cell disease']):
        return 'definition', corrected_question, corrections_made

    # Harmful/dangerous questions
    if any(word in corrected_question for word in ['harmful', 'dangerous', 'harm', 'danger']):
        return 'harmful', corrected_question, corrections_made
    
    if any(phrase in corrected_question for phrase in ['how to reduce the pains', 'reduce pain', 'pain relief', 'how to reduce pain', 'how to reduce the pains of person suffering from sickle cell', 'pain management', 'manage pain', 'how to reduce pain during crisis']):
        return 'pain_management', corrected_question, corrections_made

    # Permanent cure/getting rid of it
    if any(phrase in corrected_question for phrase in ['get rid', 'permanently', 'permanent cure', 'cure permanently','treatment']):
        return 'permanent_cure', corrected_question, corrections_made
    
    # Why does it occur
    if any(phrase in corrected_question for phrase in ['why does it occur', 'why occur', 'why happen']):
        return 'why_occur', corrected_question, corrections_made
    
    # Lifespan questions
    if any(word in corrected_question for word in ['lifespan', 'life span', 'live long', 'die', 'death', 'reduced']):
        return 'lifespan', corrected_question, corrections_made
    
    if any(phrase in corrected_question for phrase in [
    'how should i care', 
    'self care', 
    'how to manage', 
    'how can i stay healthy', 
    'lifestyle tips', 
    'daily care', 
    'prevent crisis', 
    'how to stay healthy',
    'sickle cell management'
   ]):
        return 'selfcare', corrected_question, corrections_made


    # Common disease question
    if any(phrase in corrected_question for phrase in ['common disease', 'how common', 'common condition','disease common']):
        return 'common', corrected_question, corrections_made
    
    # Difference between sickle cell disease and sickle cell anaemia
    if any(phrase in corrected_question for phrase in [
        'difference between sickle cell disease and sickle cell anaemia',
        'sickle cell disease vs sickle cell anaemia',
        'how is sickle cell disease different from sickle cell anaemia',
        'difference between sickle cell disease and sickle cell anemia'
    ]):
        return 'difference', corrected_question, corrections_made


    # Activities for improvement
    if any(phrase in corrected_question for phrase in ['activities', 'improve health', 'what to do', 'help health','exercise','activity']):
        return 'activities', corrected_question, corrections_made
    
    # What not to do/eat
    if any(phrase in corrected_question for phrase in ['what not to do', 'what not to eat', 'avoid', 'worsen', 'make worse']):
        return 'avoid', corrected_question, corrections_made
    
    # Contagious questions
    if any(word in corrected_question for word in ['contagious', 'spread', 'catch', 'transmit']):
        return 'contagious', corrected_question, corrections_made
    
    # Breastfeeding questions
    if any(phrase in corrected_question for phrase in ['feed baby', 'breastfeed', 'nursing', 'feed child']):
        return 'breastfeeding', corrected_question, corrections_made
    
    # Sexual intercourse questions
    if any(phrase in corrected_question for phrase in ['sexual intercourse', 'sex', 'intimate', 'partner infected']):
        return 'sexual', corrected_question, corrections_made
    
    # Difference from normal person
    if any(phrase in corrected_question for phrase in ['different from normal', 'difference between', 'how different']):
        return 'difference_normal', corrected_question, corrections_made
    
    # Normal activities question
    if any(phrase in corrected_question for phrase in ['normal activities', 'job', 'work', 'stress', 'sleep late','daily routine','routine']):
        return 'normal_activities', corrected_question, corrections_made
    
    # Inheritance only question
    if any(phrase in corrected_question for phrase in ['just inherited', 'inheritance only', 'inherited', 'inherit', 'is it inherited', 'is sickle cell inherited']):
        return 'inheritance', corrected_question, corrections_made

    
    # Age-related questions
    # Age-related treatment questions
    if any(phrase in corrected_question for phrase in [
    'what age', 'from what age', 'age symptoms', 'age treatment',
    'at which age', 'at what age', 'what is right age to get treatment',
    'at what age should treatment start', 'when should sickle cell treatment begin',
    'is there a best age to start treatment', 'when is the best time to treat sickle cell',
    'should treatment start early in life', 'is early treatment better for sickle cell',
    'how early should treatment be given', 'when to start meds for sickle cell',
    'can i wait to treat sickle cell', 'is there a right age for treatment',
    'too young for sickle cell treatment', 'what age do doctors recommend treatment',
    'should babies be treated for sickle cell', 'when do doctors start treating sickle cell',
    'right age', 'age to start treatment', 'when to treat',
    'best age', 'when should treatment', 'when to start treatment',
    'age should i start treatment', 'start treatment age',
    'early treatment', 'start meds', 'too young for treatment'
]):
       return 'treatment_age', corrected_question, corrections_made

    
    # Cure availability questions
    if any(phrase in corrected_question for phrase in ['everyone get cured', 'cure for everyone', 'why no cure','treatment for cure']):
        return 'cure_availability', corrected_question, corrections_made
    
    # Treatment danger questions
    if any(phrase in corrected_question for phrase in ['treatment dangerous', 'dangerous treatment', 'die from treatment']):
        return 'treatment_danger', corrected_question, corrections_made
    
    # Bone marrow danger questions
    if any(phrase in corrected_question for phrase in ['bone marrow dangerous', 'transplant dangerous', 'elder lady']):
        return 'bone_marrow_danger', corrected_question, corrections_made
    
    # Blood cell effect questions
    if any(phrase in corrected_question for phrase in ['affect blood cell', 'effect on blood', 'blood cells affected']):
        return 'blood_effect', corrected_question, corrections_made
    
    # Trait vs disease questions
    if any(phrase in corrected_question for phrase in ['trait and disease', 'trait vs disease', 'difference trait']):
        return 'trait_vs_disease', corrected_question, corrections_made
    
    # Gender inheritance questions
    if any(phrase in corrected_question for phrase in [
    'boys and girls', 'girls and boys', 'do boys and girls', 'males and females',
    'male and female', 'men and women', 'gender inherit', 'inherit equally',
    'same for boys and girls', 'is inheritance same for genders', 'will girls get it too',
    'is it equal in boys and girls', 'does gender matter', 'do both genders',
    'do boys inherit same', 'do girls inherit same', 'gender difference in inheritance',
    'gender wise inheritance'
]):
      return 'gender_inheritance', corrected_question, corrections_made

    
    # Early signs in kids
    if any(phrase in corrected_question for phrase in ['early signs', 'kids symptoms', 'children signs']):
        return 'early_signs', corrected_question, corrections_made
    
    # Untreated consequences
    if any(phrase in corrected_question for phrase in ['left untreated', 'not treated', 'ignored', 'late treatment','untreated']):
        return 'untreated', corrected_question, corrections_made
    
    # Pain crisis questions
    if any(phrase in corrected_question for phrase in ['pain crisis', 'why pain', 'pain episodes']):
        return 'pain_crisis', corrected_question, corrections_made
    
    # Organs affected
    if any(phrase in corrected_question for phrase in ['organs affected', 'which organs', 'organ damage', 'what organs', 'organs']):
        return 'organs_affected', corrected_question, corrections_made
    
    # Growth and development
    if any(phrase in corrected_question for phrase in ['growth', 'development', 'fertility', 'affect growth']):
        return 'growth_development', corrected_question, corrections_made
    
    if any(phrase in corrected_question for phrase in [
     'what is sickle cell', 
     'tell me about sickle cell', 
     'sickle cell disease meaning', 
     'define sickle cell', 
     'explain sickle cell', 
     'sickle cell anemia explanation',
     'about sickle cell'
]):
     return 'definition', corrected_question, corrections_made


    # Prenatal detection
    if any(phrase in corrected_question for phrase in ['before birth', 'prenatal', 'unborn baby', 'during pregnancy']):
        return 'prenatal', corrected_question, corrections_made
    
    # Bone marrow donor eligibility
    if any(phrase in corrected_question for phrase in ['who can give', 'donor eligible', 'bone marrow donor']):
        return 'donor_eligibility', corrected_question, corrections_made
    
    # Treatment risks
    if any(phrase in corrected_question for phrase in [
    'risks of treatment', 'treatment risks', 'side effects',
    'complications of treatment', 'is treatment safe', 'dangers of treatment',
    'hydroxyurea side effects', 'risks of hydroxyurea', 'blood transfusion risks',
    'bone marrow transplant risks', 'gene therapy risks',
    'any treatment risks', 'any side effects', 'treatment danger',
    'is there any risk in treatment', 'problems with treatment',
    'downsides of treatment', 'can treatment be harmful',
    'does treatment have complications', 'possible risks of treatment'
]):
      return 'treatment_risks', corrected_question, corrections_made

    
    # Permanent vs temporary treatment
    if any(phrase in corrected_question for phrase in ['permanent temporary', 'types of treatment', 'treatment types']):
        return 'treatment_types', corrected_question, corrections_made
    
    # Lifestyle changes
    if any(phrase in corrected_question for phrase in [
    'lifestyle changes', 'lifestyle difference', 'life with sickle cell',
    'reduce pain', 'prevent crisis', 'daily routine', 'how is their life',
    'live normal life', 'how is life different', 'affected person life',
    'precautions in daily life', 'how should i care myself', 
    'lifestyle for sickle cell', 'how do they live', 'sickle cell lifestyle', 
    'how to manage lifestyle', 'can i live normally', 'adjust life for sickle cell'
]):
      return 'lifestyle', corrected_question, corrections_made

    
    # Country statistics
    if any(phrase in corrected_question for phrase in [
    'best country for treatment', 'which country treats best', 
    'where is best treatment', 'which country is good for care',
    'best hospitals for sickle cell', 'where to get good treatment',
    'top country for treatment', 'where is treatment best', 
    'which country helps most', 'country with best healthcare for sickle cell',
    'best care for sickle cell'
]):
      return 'treatment_country', corrected_question, corrections_made

    
    # Types of sickle cell
    if any(phrase in corrected_question for phrase in ['types of sickle', 'sickle cell types', 'different types']):
        return 'sickle_types', corrected_question, corrections_made
    
    # Type detection from image
    if any(phrase in corrected_question for phrase in ['detect type', 'type from image', 'which type']):
        return 'type_detection', corrected_question, corrections_made
    
    # Blood smear information
    if any(phrase in corrected_question for phrase in ['blood smear tell', 'smear show', 'blood test']):
        return 'blood_smear', corrected_question, corrections_made
    
    # Checkup frequency
    if any(phrase in corrected_question for phrase in ['how often', 'checkup frequency', 'regular tests']):
        return 'checkup_frequency', corrected_question, corrections_made
    
    # Ayurvedic treatment
    if any(word in corrected_question for word in ['ayurvedic', 'herbal', 'traditional', 'natural']):
        return 'ayurvedic', corrected_question, corrections_made
    
    # Home remedies for crisis
    if any(phrase in corrected_question for phrase in ['home remedy', 'at home', 'pain crisis home']):
        return 'home_remedy', corrected_question, corrections_made
    
    # Hospital visit timing
    if any(phrase in corrected_question for phrase in ['when hospital', 'go to hospital', 'emergency']):
        return 'hospital_timing', corrected_question, corrections_made
    
    # Travel questions
    if any(word in corrected_question for word in ['travel', 'trip', 'vacation', 'precaution']):
        return 'travel', corrected_question, corrections_made
    
    # Safe childbearing
    if any(phrase in corrected_question for phrase in ['have children', 'safe pregnancy', 'children safely']):
        return 'safe_childbearing', corrected_question, corrections_made
    
    # Doctor/specialist questions
    if any(phrase in corrected_question for phrase in ['which doctor', 'specialist', 'hematologist', 'best doctor']):
        return 'doctor_specialist', corrected_question, corrections_made
    
    if any(phrase in corrected_question for phrase in [
    'what is anemia', 
    'tell me about anemia', 
    'explain anemia', 
    'define anemia', 
    'anaemia',  # UK spelling
    'sickle cell anemia',
    'anemia meaning'
]):
     return 'anemia', corrected_question, corrections_made


    # Treatment locations
    if any(phrase in corrected_question for phrase in [
    'best treatment locations', 'where to get treated', 'treatment centers',
    'specialist hospitals for sickle cell', 'best hospitals for sickle cell',
    'top clinics for treatment', 'where should i get treated', 
    'treatment facilities', 'where to go for sickle cell treatment',
    'sickle cell centers', 'famous treatment hospitals', 
    'locations for sickle cell care', 'best place to cure sickle cell',
    'best hospitals for bone marrow transplant'
]):
      return 'treatment_locations', corrected_question, corrections_made

    # Prevention questions
    if any(phrase in corrected_question for phrase in [
    'how to prevent', 
    'prevention', 
    'avoid getting', 
    'stop from getting sickle cell', 
    'prevent sickle cell', 
    'prevent disease'
]):
      return 'prevention', corrected_question, corrections_made

    # Pain management queries
    if any(phrase in corrected_question for phrase in [
    'pain management', 'manage pain at home', 'pain relief options',
    'alternative pain management', 'non-drug pain relief', 'chronic pain'
]):
     return 'pain_management', corrected_question, corrections_made

# Mental Health & Emotional Support
    if any(phrase in corrected_question for phrase in [
    'mental health', 'depression', 'anxiety', 'coping strategies',
    'emotional support', 'support groups', 'psychological help','life is ruined', 'i feel hopeless', 'why only me', 'i am scared'
]):
     return 'mental_health', corrected_question, corrections_made

# Financial & Insurance Questions
    if any(phrase in corrected_question for phrase in [
    'insurance coverage', 'treatment cost', 'financial assistance',
    'medication cost', 'affordable care', 'government programs'
]):
     return 'financial', corrected_question, corrections_made

# School & Education Accommodations
    if any(phrase in corrected_question for phrase in [
    'school accommodations', 'education support', 'special needs at school',
    'college with sickle cell', 'academic help', 'school rights'
]):
     return 'education', corrected_question, corrections_made

# Employment & Workplace Rights
    if any(phrase in corrected_question for phrase in [
    'workplace rights', 'job discrimination', 'reasonable accommodations',
    'disability benefits', 'sick leave', 'career with sickle cell'
]):
     return 'employment', corrected_question, corrections_made

# Emergency Preparedness
    if any(phrase in corrected_question for phrase in [
    'emergency kit', 'crisis preparation', 'what to keep at home',
    'hospital bag', 'emergency contacts', 'disaster planning'
]):
      return 'emergency_prep', corrected_question, corrections_made

# Alternative & Complementary Therapies
    if any(phrase in corrected_question for phrase in [
    'acupuncture', 'massage therapy', 'meditation', 'yoga for pain',
    'chiropractic care', 'complementary medicine'
]):
      return 'alternative_therapies', corrected_question, corrections_made

# Transition from Pediatric to Adult Care
    if any(phrase in corrected_question for phrase in [
    'transition to adult care', 'pediatric to adult doctor',
    'changing doctors as adult', 'adult hematologist'
]):
     return 'care_transition', corrected_question, corrections_made

# Clinical Trials & Research Participation
    if any(phrase in corrected_question for phrase in [
    'clinical trials', 'research studies', 'experimental treatments',
    'participate in research', 'new treatments available'
]):
     return 'clinical_trials', corrected_question, corrections_made

# Sickle Cell in Older Adults
    if any(phrase in corrected_question for phrase in [
    'sickle cell in elderly', 'aging with sickle cell', 'older patients',
    'seniors with sickle cell', 'geriatric sickle cell care'
]):
     return 'aging', corrected_question, corrections_made


    # ORIGINAL QUESTION TYPES (keeping existing logic)
    # Question categories - check food/diet questions FIRST before treatment
    if any(word in corrected_question for word in ['food', 'eat', 'diet', 'nutrition', 'meal', 'drink']):
        return 'diet', corrected_question, corrections_made
    
    if any(word in corrected_question for word in ['highlight', 'show', 'display', 'mark', 'identify']):
        if any(word in corrected_question for word in ['sickle', 'abnormal', 'affected']):
            return 'highlight', corrected_question, corrections_made
    
    if any(word in corrected_question for word in ['symptoms', 'symptom', 'signs']):
        return 'symptoms', corrected_question, corrections_made
    
    if 'what is' in corrected_question and any(word in corrected_question for word in ['sickle', 'anemia', 'anaemia']):
        return 'definition', corrected_question, corrections_made
    
    if any(word in corrected_question for word in ['how', 'why']) and any(word in corrected_question for word in ['get', 'cause', 'develop', 'form']):
        return 'causes', corrected_question, corrections_made
    
    if any(word in corrected_question for word in ['dangerous', 'severe', 'risk']):
        return 'severity', corrected_question, corrections_made
    
    if any(word in corrected_question for word in ['stages', 'stage', 'levels']):
        return 'stages', corrected_question, corrections_made
    
    if any(word in corrected_question for word in ['treatment', 'treat', 'cure', 'therapy']):
        return 'treatment', corrected_question, corrections_made
    
    if any(word in corrected_question for word in ['genetic', 'hereditary', 'inherited']):
        return 'genetic', corrected_question, corrections_made
    
    if any(word in corrected_question for word in ['shape', 'look', 'appearance']):
        return 'shape', corrected_question, corrections_made
    
    if any(word in corrected_question for word in ['normal', 'abnormal', 'healthy', 'infected']):
        return 'status', corrected_question, corrections_made
    
    if any(word in corrected_question for word in ['red', 'orange', 'boxes', 'highlighted']):
        return 'boxes', corrected_question, corrections_made
    
    return 'general', corrected_question, corrections_made

# Enhanced generate natural responses based on prediction and question type
def generate_response(question_type, prediction_label, confidence, corrected_question, corrections_made):
    is_sickle = prediction_label == "Sickle Cell"
    
    response = ""
    
    # Add spelling correction notice if needed - only show specific corrections
    if corrections_made:
        corrections_text = ", ".join(corrections_made)
        response += f"🔧 Spelling corrections: {corrections_text}\n\n"
    
    # NEW RESPONSE CASES
    

    if question_type == 'image_content':
     response += "Looking at this blood smear image, I can see:\n\n"
     response += "**What's in the image:**\n"
     response += "• Red blood cells (RBCs) - the main components visible\n"
     response += "• Blood plasma (the background)\n"
     if is_sickle:
        response += f"• **Red blood cells including sickle-shaped cells** - I can detect sickle cells with {confidence*100:.1f}% confidence\n"
        response += "• Some cells show the characteristic crescent or banana shape (sickle cells)\n"
        response += "• The sickle cells appear elongated and rigid compared to normal round cells\n"
        response += "• Both normal and abnormal (sickle) cells are present\n\n"
        response += "This blood smear shows signs of sickle cell disease."
     else:
        response += f"• **Normal, healthy red blood cells** - appearing round and disc-shaped as they should be\n"
        response += f"• No abnormal or sickle-shaped cells detected (I'm {confidence*100:.1f}% confident these are normal cells)\n\n"
        response += "This appears to be a healthy blood sample with normal red blood cells."
        return response, False

    if question_type == 'color_analysis':
        response += "🎨 **RBC Color & Stain Analysis Request**\n\n"
        response += "For detailed color analysis and stain identification, I'll open our specialized color analysis tool for you!\n\n"
        response += "**What the color analysis tool will do:**\n"
        response += "• 🔍 Detect the exact color of your RBCs (Blue, Red, Purple, Green, etc.)\n"
        response += "• 🧪 Identify the most likely stain used with confidence scores\n"
        response += "• 📊 Provide detailed RGB values and statistical analysis\n"
        response += "• 📋 Generate a downloadable analysis report\n\n"
        response += "**Supported stains include:**\n"
        response += "• Giemsa, Wright's, Leishman stains\n"
        response += "• H&E, Methylene Blue, Trypan Blue\n"
        response += "• Gram stain, Crystal Violet, and more!\n\n"
        response += "Click the button below to open the color analysis tool! 👇"
        return response, True  #
    
    elif question_type == 'harmful':
        response += "**Is sickle cell disease harmful or dangerous?**\n\n"
        response += "Sickle cell disease can be serious, but the level of harm varies greatly:\n\n"
        response += "**Potential complications:**\n"
        response += "• Painful episodes (pain crises)\n• Organ damage over time\n• Increased risk of infections\n• Stroke risk\n• Acute chest syndrome\n• Kidney problems\n\n"
        response += "**However, with proper medical care:**\n"
        response += "• Many people live normal, productive lives\n• Modern treatments significantly reduce complications\n• Early detection and management prevent many problems\n• Most people can work, study, and have families\n\n"
        response += "**The key is:** Don't ignore it - get proper medical care and follow treatment plans."
        if is_sickle:
            response += f"\n\nSince your blood sample shows signs of sickle cell disease ({confidence*100:.1f}% confidence), it's important to work with a hematologist for proper management."
        return response, False

    elif question_type == 'definition':
        response += "**What is Sickle Cell Disease?**\n\n"
        response += "Sickle cell disease is a **genetic blood disorder** that affects the shape and function of red blood cells.\n\n"
        response += "**Normal red blood cells** are round and flexible, allowing them to move easily through blood vessels.\n"
        response += "**In sickle cell disease**, the red blood cells become hard, sticky, and shaped like a sickle or crescent moon.\n\n"
        response += "**This causes problems such as:**\n"
        response += "• Blocking blood flow (pain and organ damage)\n"
        response += "• Shorter red blood cell lifespan (causing anemia)\n"
        response += "• Higher risk of infection and stroke\n\n"
        response += "**Key facts:**\n"
        response += "• It's inherited — you must get the gene from both parents\n"
        response += "• It affects millions worldwide, especially in Africa, India, and the Middle East\n"
        response += "• While there's no universal cure, treatments can reduce symptoms and improve life expectancy"
        return response, False
    
    if question_type == 'causes':
        response += "**What causes sickle cell disease?**\n\n"
        response += "• Sickle cell disease is caused by a change (mutation) in the **HBB gene**, which makes part of the hemoglobin protein in red blood cells.\n"
        response += "• Specifically, a single-letter change in the DNA causes the amino acid **valine** to replace **glutamic acid** at position 6 of the beta-globin chain — this is called the **HbS mutation**.\n"
        response += "• When a person has two copies of this mutated gene (one from each parent), they develop sickle cell disease. If they have only one copy, they are a **carrier** (sickle cell trait) and usually have milder or no symptoms.\n\n"
        response += "• The mutated hemoglobin makes red blood cells stiff and sickle-shaped under low-oxygen or stressed conditions, causing blockages in small blood vessels, pain crises, and other complications.\n\n"
        response += "🔎 **Summary:** It's a hereditary genetic mutation — not caused by infection, diet, or lifestyle. Genetic counseling can help families understand the risk of passing it to children."
        return response, False
    
    if question_type == 'definition':
        response += "**What is Sickle Cell Disease / Sickle Cell Anaemia?**\n\n"
        response += "• **Sickle Cell Disease (SCD)** is a group of inherited blood disorders caused by abnormal hemoglobin (mostly the HbS mutation).\n"
        response += "• **Sickle Cell Anaemia** is a common and often-severe form of SCD (usually when a person has two HbS genes, often called HbSS).\n\n"
        response += "Key points:\n"
        response += "• Normal red blood cells are flexible and round; in SCD they can become rigid and crescent-shaped (sickle-shaped).\n"
        response += "• These sickled cells can block small blood vessels → pain episodes, tissue/organ damage, and anemia (because sickled cells break down sooner).\n"
        response += "• It's inherited: you get one gene from each parent. Two abnormal copies = disease; one abnormal copy = trait (carrier).\n\n"
        response += "⚠️ Note: The terms are often used interchangeably in casual speech, but 'sickle cell disease' is the broader term for the genetic condition; 'sickle cell anaemia' commonly refers to the HbSS subtype."
        return response, False
    
    if question_type == 'pain_management':
        response += "**How to reduce pain for someone with sickle cell (practical measures)**\n\n"
        response += "Immediate/home measures:\n"
        response += "• **Hydration:** Drink plenty of fluids — dehydration can trigger or worsen pain crises.\n"
        response += "• **Warmth:** Use warm compresses or heating pads (not too hot) to relax muscles and improve circulation.\n"
        response += "• **Rest:** Minimize activity and rest the painful area.\n"
        response += "• **Over-the-counter pain relief:** Paracetamol (acetaminophen) or ibuprofen can help for mild pain (follow dosing guidance).\n\n"
        response += "Medical treatments (see a doctor if pain is severe or not improving):\n"
        response += "• **Prescription analgesics:** Stronger pain medicines (opioids) may be used under supervision during severe crises.\n"
        response += "• **IV fluids & oxygen:** In hospital, IV fluids and supplemental oxygen help during crises.\n"
        response += "• **Blood transfusion:** Used for severe complications (e.g., acute chest syndrome, stroke prevention).\n"
        response += "• **Disease-modifying drugs:** **Hydroxyurea** reduces frequency of painful episodes for many patients; new medications and gene therapies are also emerging.\n\n"
        response += "Prevention & trigger avoidance:\n"
        response += "• Avoid extreme temperatures, high altitudes, and dehydration.\n"
        response += "• Treat infections early and keep vaccinations up to date.\n"
        response += "• Follow regular care with a hematologist and have an individualized crisis plan.\n\n"
        response += "Important: If pain is severe, increasing, or accompanied by fever, difficulty breathing, chest pain, or neurological symptoms — seek emergency care immediately.\n\n"
        response += "🩺 I’m not a doctor — encourage the person to have a treatment plan from their hematologist. For persistent or severe pain, urgent medical evaluation is required."
        return response, False
    

    elif question_type == 'prevention':
        response += "**How to prevent sickle cell disease**\n\n"
        response += "Sickle cell disease is a **genetic condition**, so it cannot be prevented in the traditional sense.\n\n"
        response += "**Key prevention strategies for families:**\n"
        response += "• **Genetic counseling:** If both parents carry the sickle cell trait, counseling can help understand the risk to children.\n"
        response += "• **Prenatal testing:** Testing during pregnancy can detect if the baby has sickle cell disease.\n"
        response += "• **Awareness and family planning:** Knowing carrier status helps make informed decisions.\n\n"
        response += "**Healthy lifestyle steps for carriers or patients:**\n"
        response += "• Stay hydrated and avoid triggers for sickle cell crises\n"
        response += "• Maintain a balanced diet and follow medical advice\n"
        response += "• Regular checkups to monitor health\n\n"
        response += "⚠️ Important: While you can’t prevent inheriting sickle cell if both parents carry the gene, early detection and proper care prevent complications."
        return response, False
    
    elif question_type == 'pain_management':
        response += "**Pain Management for Sickle Cell Disease**\n\n"
        response += "Managing pain is a key part of living with sickle cell disease:\n"
        response += "• Take prescribed medications as advised by your doctor\n"
        response += "• Use heat packs or warm baths to relieve pain\n"
        response += "• Gentle exercise can help reduce stiffness\n"
        response += "• Stay hydrated to prevent pain crises\n"
        response += "• Consider relaxation techniques like deep breathing, meditation, or yoga\n"
        response += "• For chronic pain, consult a pain specialist for personalized management\n"
        if is_sickle:
            response += f"\n⚠️ Since your blood sample shows signs of sickle cells ({confidence*100:.1f}% confidence), careful monitoring and medical guidance are important."
        return response, False

    elif question_type == 'mental_health':
        response += "**Mental Health & Emotional Support**\n\n"
        response += "Living with sickle cell can affect emotional well-being:\n"
        response += "• Feelings of anxiety or depression are common\n"
        response += "• Talking to a counselor or psychologist can help\n"
        response += "• Join support groups to connect with others who understand\n"
        response += "• Mindfulness, meditation, and stress-reduction exercises are beneficial\n"
        return response, False

    elif question_type == 'financial':
        response += "**Financial & Insurance Guidance**\n\n"
        response += "Managing the cost of sickle cell treatment:\n"
        response += "• Check if your health insurance covers medications and doctor visits\n"
        response += "• Explore government programs or financial assistance for treatments\n"
        response += "• Discuss generic medications or hospital payment plans with your healthcare provider\n"
        return response, False

    elif question_type == 'education':
        response += "**School & Education Support**\n\n"
        response += "Tips for students with sickle cell disease:\n"
        response += "• Ask for accommodations like extra rest periods or flexible deadlines\n"
        response += "• Inform teachers or school nurses about your condition\n"
        response += "• Consider individualized education plans (IEPs) if needed\n"
        return response, False

    elif question_type == 'employment':
        response += "**Workplace & Employment Guidance**\n\n"
        response += "Managing sickle cell at work:\n"
        response += "• Know your rights for sick leave and reasonable accommodations\n"
        response += "• Discuss flexible schedules with your employer if needed\n"
        response += "• Disability benefits may be available if your condition impacts work\n"
        return response, False

    elif question_type == 'emergency_prep':
        response += "**Emergency Preparedness**\n\n"
        response += "Preparing for a sickle cell crisis:\n"
        response += "• Keep an emergency kit with medications, hydration, and contact info\n"
        response += "• Have a hospital bag ready for unexpected crises\n"
        response += "• Inform family and friends about emergency contacts and care steps\n"
        return response, False

    elif question_type == 'alternative_therapies':
        response += "**Alternative & Complementary Therapies**\n\n"
        response += "Some patients benefit from complementary approaches:\n"
        response += "• Acupuncture, massage therapy, and yoga can reduce pain and stress\n"
        response += "• Meditation and relaxation techniques help with mental well-being\n"
        response += "• Always discuss any alternative therapy with your doctor before starting"
        return response, False

    elif question_type == 'care_transition':
        response += "**Transition from Pediatric to Adult Care**\n\n"
        response += "Moving from a pediatric hematologist to adult care:\n"
        response += "• Schedule a transition plan with both pediatric and adult doctors\n"
        response += "• Keep records of past treatments and medications\n"
        response += "• Learn to manage your appointments and medications independently"
        return response, False

    elif question_type == 'clinical_trials':
        response += "**Clinical Trials & Research Participation**\n\n"
        response += "Opportunities to participate in research or experimental treatments:\n"
        response += "• Clinical trials may provide access to new therapies\n"
        response += "• Discuss eligibility and potential risks with your doctor\n"
        response += "• Check trusted sources like ClinicalTrials.gov for ongoing studies"
        return response, False

    elif question_type == 'aging':
        response += "**Sickle Cell in Older Adults**\n\n"
        response += "Managing sickle cell as you age:\n"
        response += "• Older adults may face additional complications\n"
        response += "• Regular checkups are important for organ health\n"
        response += "• Adapt lifestyle and treatment plans as needed for age-related changes"
        return response, False


    elif question_type == 'anemia':
        response += "**What is Anemia?**\n\n"
        response += "Anemia is a condition where you **don’t have enough healthy red blood cells or hemoglobin** to carry oxygen to your body’s tissues.\n\n"
        response += "**Symptoms of anemia can include:**\n"
        response += "• Fatigue or weakness\n"
        response += "• Pale or yellowish skin\n"
        response += "• Shortness of breath\n"
        response += "• Dizziness or headaches\n"
        response += "• Cold hands and feet\n\n"
        response += "**Types of anemia include:**\n"
        response += "• Iron-deficiency anemia (most common)\n"
        response += "• Vitamin B12 or folate deficiency anemia\n"
        response += "• Sickle cell anemia (a genetic form)\n"
        response += "• Aplastic anemia (due to bone marrow problems)\n\n"
        response += "**Key facts:**\n"
        response += "• It’s often caused by poor diet, blood loss, or chronic diseases\n"
        response += "• It can be treated based on the cause — diet changes, supplements, or medical care\n"
        response += "• Sickle cell anemia is a type of inherited anemia where red blood cells are misshapen and break down easily"
        return response, False


    elif question_type == 'diet':
      response += "**Diet and Nutrition for Sickle Cell Disease**\n\n"
      response += "A healthy, balanced diet can help reduce symptoms and prevent complications:\n\n"
      response += "**Recommended:**\n"
      response += "• High-folate foods (e.g., leafy greens)\n"
      response += "• Iron-rich foods if not contraindicated\n"
      response += "• Lots of water to prevent dehydration\n"
      response += "• Fruits, vegetables, whole grains\n\n"
      response += "**Avoid or Limit:**\n"
      response += "• Alcohol\n• High-sugar processed foods\n• Smoking\n\n"
      response += "**Key message:** Nutrition alone won't cure sickle cell, but it strengthens your immune system and improves overall health."
      return response, False

    elif question_type == 'permanent_cure':
        response += "**Can you get rid of sickle cell disease permanently?**\n\n"
        response += "Currently, there is only **one potential permanent cure:**\n\n"
        
        response += "**Bone Marrow (Stem Cell) Transplant:**\n"
        response += "• Can potentially cure sickle cell disease completely\n"
        response += "• Success rate is quite high when a good donor is found\n"
        response += "• However, it's not suitable for everyone\n"
        response += "• Requires a compatible donor (often a sibling)\n"
        response += "• Has risks and complications\n"
        response += "• Best results in younger patients (under 16)\n\n"
        
        response += "**New promising treatments:**\n"
        response += "• Gene therapy - still experimental but showing promise\n"
        response += "• Gene editing (CRISPR) - in clinical trials\n"
        response += "• These may become available in the future\n\n"
        
        response += "**For now:** Most people manage the disease very well with medications like hydroxyurea, which can make symptoms much milder, even though it doesn't cure the disease.\n\n"

        response += "---\n\n"
        response += "🔹 **Important Note:** There is currently no universal cure, but treatments can manage symptoms and improve quality of life:\n"
        response += "- Hydroxyurea: reduces painful crises and need for transfusions\n"
        response += "- Blood transfusions: prevent stroke and reduce anemia\n"
        response += "- Bone marrow or stem cell transplant: potential cure in select patients, usually children\n"
        response += "- Pain management with medications\n"
        response += "- Vaccinations and antibiotics to prevent infections\n"
        response += "- Oxygen therapy in some cases\n"

        return response, False

    elif question_type == 'difference':
        response += "**Difference between Sickle Cell Disease and Sickle Cell Anaemia:**\n\n"
        response += "• **Sickle Cell Disease (SCD):** A genetic disorder affecting hemoglobin that can include multiple sickle cell conditions.\n"
        response += "• **Sickle Cell Anaemia (SCA):** The most common and severe type of sickle cell disease, caused specifically by inheriting two sickle hemoglobin genes (HbSS).\n"
        response += "• **Summary:** All SCA is SCD, but not all SCD is SCA."
        return response, False


    elif question_type == 'why_occur':
        response += "**Why does sickle cell disease occur?**\n\n"
        response += "It's purely **genetic** - here's the simple explanation:\n\n"
        response += "**The root cause:**\n"
        response += "• A mutation in the gene that makes hemoglobin (the protein that carries oxygen)\n• This mutated gene is passed down from parents to children\n• When you inherit the sickle cell gene from BOTH parents, you get the disease\n• If you get it from only ONE parent, you have sickle cell trait (usually harmless)\n\n"
        response += "**Why the cells become sickle-shaped:**\n"
        response += "• The mutated hemoglobin forms long, rigid chains when oxygen levels drop\n• These chains distort the normally round, flexible red blood cells\n• The cells become crescent or 'sickle' shaped\n• These rigid cells can block blood flow and break apart easily\n\n"
        response += "**Important:** You cannot develop sickle cell disease from lifestyle, diet, or environment - you're born with it."
        return response, False

    elif question_type == 'lifespan':
        response += "**Will lifespan be reduced with sickle cell disease?**\n\n"
        response += "The good news is that life expectancy has improved dramatically:\n\n"
        response += "**Current outlook:**\n"
        response += "• People born today with sickle cell disease often live into their 50s, 60s, and beyond\n• Some live completely normal lifespans\n• With proper medical care, many complications can be prevented\n\n"
        response += "**Factors that affect lifespan:**\n"
        response += "• **Quality of medical care** (most important factor)\n• Early diagnosis and treatment\n• Following treatment plans consistently\n• Healthy lifestyle choices\n• Access to preventive care\n\n"
        response += "**Historical vs. Modern:**\n"
        response += "• In the past (1970s), average lifespan was much shorter\n• Modern medicine has changed this dramatically\n• New treatments continue to improve outcomes\n\n"
        response += "**Key message:** With proper care, most people with sickle cell disease can expect to live long, fulfilling lives."
        return response, False

    elif question_type == 'common':
        response += "**Is sickle cell disease a common condition?**\n\n"
        response += "**Global perspective:**\n"
        response += "• Affects millions of people worldwide\n• Most common in people with ancestry from Africa, Mediterranean, Middle East, and parts of India\n• About 1 in 365 African American babies are born with sickle cell disease\n• About 1 in 16,300 Hispanic American babies\n\n"
        response += "**Why it's more common in certain regions:**\n"
        response += "• The sickle cell gene provides protection against malaria\n• In areas where malaria is common, the gene persisted because it helped people survive malaria\n• This is why it's more common in people from tropical regions\n\n"
        response += "**In summary:** It's common in certain ethnic groups but relatively rare in the general population of some countries. However, due to global migration, it's now found worldwide."
        return response, False

    elif question_type == 'activities':
        response += "**Activities to improve health with sickle cell disease:**\n\n"
        response += "**Physical Activities (with caution):**\n"
        response += "• Light to moderate exercise (walking, swimming, yoga)\n• Avoid intense, prolonged exercise\n• Stay well-hydrated during any activity\n• Rest when you feel tired\n\n"
        response += "**Daily Health Practices:**\n"
        response += "• Drink 8-10 glasses of water daily\n• Take prescribed medications regularly\n• Get adequate sleep (7-9 hours)\n• Eat a balanced, nutritious diet\n• Take folic acid supplements as prescribed\n\n"
        response += "**Preventive Measures:**\n"
        response += "• Get regular medical check-ups\n• Stay up-to-date with vaccinations\n• Avoid extreme temperatures (too hot or cold)\n• Manage stress through relaxation techniques\n• Avoid smoking and excessive alcohol\n\n"
        response += "**Mental Health:**\n"
        response += "• Stay connected with family and friends\n• Consider support groups\n• Practice stress management\n• Maintain hobbies and interests you enjoy"
        return response, False

    elif question_type == 'avoid':
        response += "**What NOT to do and eat that could worsen sickle cell health:**\n\n"
        response += "**❌ Activities to AVOID:**\n"
        response += "• Extreme physical exertion or intense exercise\n• Getting dehydrated (most important!)\n• Exposure to extreme cold or heat\n• High altitudes (like mountain climbing)\n• Smoking or using tobacco\n• Excessive alcohol consumption\n• Ignoring pain or symptoms\n\n"
        response += "**🚫 Foods/Drinks to LIMIT:**\n"
        response += "• Excessive caffeine (can cause dehydration)\n• Too much alcohol (triggers dehydration and pain)\n• High-sodium processed foods\n• Sugary drinks that don't hydrate well\n\n"
        response += "**⚠️ Situations to be careful with:**\n"
        response += "• Flying without staying hydrated\n• Swimming in very cold water\n• Skipping meals regularly\n• Not taking prescribed medications\n• Stress without management techniques\n\n"
        response += "**💡 Remember:** The #1 trigger for sickle cell crises is dehydration, so always prioritize staying well-hydrated!"
        return response, False

    elif question_type == 'contagious':
        response += "**Is sickle cell disease contagious?**\n\n"
        response += "**Absolutely NOT!** Sickle cell disease is **NOT contagious** at all.\n\n"
        response += "**You CANNOT catch it from:**\n"
        response += "• Being around someone who has it\n• Sharing food, drinks, or utensils\n• Hugging, kissing, or touching\n• Sexual contact\n• Blood contact (it's genetic, not infectious)\n• Coughing, sneezing, or breathing the same air\n\n"
        response += "**Why it's not contagious:**\n"
        response += "• It's a genetic condition - you're born with it\n• It's caused by a gene mutation, not by bacteria or viruses\n• You inherit it from your parents through their genes\n\n"
        response += "**What this means:**\n"
        response += "• It's completely safe to be around people with sickle cell disease\n• They can live normally in families, schools, and workplaces\n• No special precautions needed to prevent 'catching' it\n\n"
        response += "**The only way to 'get' sickle cell disease is to inherit the genes from both parents at birth.**"
        return response, False

    elif question_type == 'breastfeeding':
        response += "**Can a woman with sickle cell disease breastfeed her baby?**\n\n"
        response += "**Yes, absolutely!** Women with sickle cell disease can safely breastfeed their babies.\n\n"
        response += "**Benefits for the baby:**\n"
        response += "• Breast milk provides excellent nutrition\n• Antibodies in breast milk help protect the baby from infections\n• This is especially important since babies with sickle cell disease have higher infection risk\n\n"
        response += "**Important considerations for the mother:**\n"
        response += "• Stay very well-hydrated (even more than usual)\n• Eat a nutritious diet to maintain energy\n• Get adequate rest when possible\n• Continue taking prescribed medications (most are safe during breastfeeding)\n• Monitor for signs of fatigue or pain crises\n\n"
        response += "**Special notes:**\n"
        response += "• Breastfeeding itself doesn't worsen sickle cell disease\n• However, the physical demands of caring for a newborn can be tiring\n• Having support from family/partners is very helpful\n• Consult with your hematologist about any medications\n\n"
        response += "**Bottom line:** Breastfeeding is encouraged and safe for both mother and baby!"
        return response, False

    elif question_type == 'sexual':
        response += "**Can you have sexual intercourse if you have sickle cell disease?**\n\n"
        response += "**Yes, absolutely!** Having sickle cell disease doesn't prevent you from having a normal sexual life.\n\n"
        response += "**Important considerations:**\n"
        response += "• Sickle cell disease is NOT sexually transmitted\n• You cannot give it to your partner through sexual contact\n• It's completely safe for both you and your partner\n\n"
        response += "**Things to keep in mind:**\n"
        response += "• Stay well-hydrated before and after\n• Don't overexert yourself if you're feeling unwell\n• Communication with your partner is important\n• If you're planning to have children, consider genetic counseling\n\n"
        response += "**Fertility considerations:**\n"
        response += "• Most people with sickle cell disease can have children\n• Genetic counseling can help understand risks to future children\n• Pregnancy may need extra medical monitoring\n\n"
        response += "**Bottom line:** Your romantic and sexual life can be completely normal!"
        return response, False

    elif question_type == 'difference_normal':
        response += "**How is a sickle cell infected person different from a normal person?**\n\n"
        response += "**Physical differences:**\n"
        response += "• Red blood cells are crescent-shaped instead of round\n• These abnormal cells can block blood flow\n• May experience periodic pain episodes (crises)\n• Might get tired more easily during flare-ups\n• Higher risk of infections\n\n"
        response += "**Daily life differences:**\n"
        response += "• Need to drink more water (stay very hydrated)\n• May need to avoid extreme temperatures\n• Require regular medical check-ups\n• Take daily medications (like hydroxyurea)\n• Need to be more careful about infections\n\n"
        response += "**What's the SAME as normal people:**\n"
        response += "• Intelligence and mental abilities are completely normal\n• Can work, study, and have careers\n• Can have relationships and families\n• Can enjoy hobbies and social activities\n• Life goals and dreams remain the same\n\n"
        response += "**Key point:** The main difference is in managing health - most aspects of life remain completely normal!"
        return response, False

    elif question_type == 'normal_activities':
        response += "**Can a person with sickle cell do normal activities like work, handle stress, or sleep late?**\n\n"
        response += "**Work/Job:**\n"
        response += "• ✅ Yes, most people with sickle cell can work normally\n• Choose jobs that don't involve extreme physical exertion\n• Office work, teaching, healthcare, business - all fine\n• May need occasional sick days during pain crises\n• Employers legally cannot discriminate based on sickle cell status\n\n"
        response += "**Handling Stress:**\n"
        response += "• ⚠️ Need to manage stress more carefully\n• High stress can trigger pain crises\n• Practice stress management techniques (meditation, deep breathing)\n• Exercise lightly to reduce stress\n• Seek support when overwhelmed\n\n"
        response += "**Sleep Schedule:**\n"
        response += "• ⚠️ Try to avoid regularly sleeping very late\n• Adequate sleep (7-9 hours) is important for health\n• Irregular sleep can weaken immune system\n• Occasionally staying up late is okay, but not as a habit\n\n"
        response += "**Other Normal Activities:**\n"
        response += "• Social gatherings - ✅ Yes\n• Sports (moderate) - ✅ Yes\n• Travel - ✅ Yes (with precautions)\n• Education - ✅ Completely normal\n\n"
        response += "**Bottom line:** You can live a very normal life with some smart health management!"
        return response, False

    elif question_type == 'inheritance':
        response += "**Is sickle cell  inherited?**\n\n"
        response += "**Yes, sickle cell disease is ONLY inherited - you cannot develop it any other way.**\n\n"
        response += "**How inheritance works:**\n"
        response += "• You inherit genes from BOTH parents\n• Each parent gives you one copy of the hemoglobin gene\n• **To have sickle cell DISEASE:** You need the sickle gene from BOTH parents\n• **To have sickle cell TRAIT:** You get the sickle gene from only ONE parent\n\n"
        response += "**Different scenarios:**\n"
        response += "• **Both parents have sickle cell disease:** All children will have the disease\n• **Both parents have sickle cell trait:** 25% chance each child gets the disease\n• **One parent has disease, one has trait:** 50% chance each child gets the disease\n• **One parent has trait, one is normal:** Children can only get trait (not disease)\n• **Both parents are normal:** Children cannot have sickle cell disease\n\n"
        response += "**What you CANNOT get sickle cell from:**\n"
        response += "• Poor diet or lifestyle\n• Infections or diseases\n• Environmental factors\n• Injuries or accidents\n• Contact with affected people\n\n"
        response += "**Key message:** It's purely genetic - determined at the moment of conception!"
        return response, False

    elif question_type == 'treatment_age':
        response += "**🧒 What is the Right Age to Start Treatment for Sickle Cell Disease?**\n\n"
        response += "Treatment should start **as early as possible**, ideally in **infancy**, immediately after diagnosis. Early treatment greatly improves long-term health outcomes and prevents complications.\n\n"

        response += "**🍼 Newborn to 6 months:**\n"
        response += "• Begin medical monitoring right after birth\n"
        response += "• Start **penicillin** (usually by 2 months) to prevent infections\n"
        response += "• **Folic acid supplements** to support red blood cell production\n"
        response += "• Follow standard **vaccination schedules** closely\n\n"

        response += "**👶 6 months and above:**\n"
        response += "• Continue antibiotics and vaccinations\n"
        response += "• **Hydroxyurea** may be prescribed to reduce complications\n"
        response += "• Implement pain management strategies\n\n"

        response += "**🏥 Advanced/Permanent Treatment Options:**\n"
        response += "• **Bone marrow/stem cell transplant:** Best outcomes in children under 16\n"
        response += "• Can still be performed in adults if a suitable donor is available\n\n"

        response += "**💡 Why Early Treatment Matters:**\n"
        response += "• Prevents organ damage and infections\n"
        response += "• Supports healthy growth and development\n"
        response += "• Reduces pain crises and hospitalization\n\n"

        response += "**✅ Key Takeaways:**\n"
        response += "• Start treatment **as soon as sickle cell is diagnosed**\n"
        response += "• There is **no 'perfect' age — earlier is better**\n"
        response += "• Treatment at **any age** is beneficial\n"
        response += "• **Younger age = better outcomes**, especially for advanced procedures"
        return response, False

    elif question_type == 'cure_availability':
        response += "**Can everyone get cured if treated? Why isn't there a cure for everyone?**\n\n"
        response += "**Unfortunately, not everyone can be cured currently. Here's why:**\n\n"
        response += "**Bone Marrow Transplant (the main cure):**\n"
        response += "• Only 20-30% of patients have a suitable donor\n• Best donors are siblings with matching tissue types\n• Risk of serious complications\n• Very expensive and requires specialized centers\n• Age matters - better outcomes in younger patients\n\n"
        response += "**Why not everyone can get it:**\n"
        response += "• **No matching donor** (most common reason)\n• Patient too old or too sick for the procedure\n• Other health conditions that make it risky\n• Limited availability of specialized treatment centers\n• Cost and insurance coverage issues\n\n"
        response += "**Good news - new treatments coming:**\n"
        response += "• Gene therapy - showing promise in trials\n• Gene editing (CRISPR) - being tested\n• Better medications being developed\n• These may help more people in the future\n\n"
        response += "**Current reality:**\n"
        response += "• Most people manage very well with medications\n• Quality of life has improved dramatically\n• Research continues for better cures\n\n"
        response += "**Hope:** Medical advances are making cures available to more people each year!"
        return response, False

    elif question_type == 'treatment_danger':
        response += "**Is sickle cell treatment dangerous? Could you die from treatment?**\n\n"
        response += "**Daily treatments (medications) are generally very safe:**\n\n"
        response += "**Hydroxyurea (most common medicine):**\n"
        response += "• Very safe when monitored by doctors\n• Side effects are usually mild\n• Regular blood tests ensure safety\n• Benefits far outweigh risks\n\n"
        response += "**Pain medications:**\n"
        response += "• Generally safe when used as prescribed\n• Doctors monitor for any issues\n\n"
        response += "**Bone Marrow Transplant (permanent cure):**\n"
        response += "• ⚠️ This does have more serious risks\n• Small risk of death (about 5-10% in experienced centers)\n• Risk of graft-versus-host disease\n• Risk of infections during recovery\n• BUT - most people do very well\n\n"
        response += "**Important perspective:**\n"
        response += "• Risk of NOT treating sickle cell is much higher than treatment risks\n• Untreated sickle cell can cause organ damage and early death\n• Modern treatments are much safer than in the past\n• Experienced medical teams minimize risks\n\n"
        response += "**Bottom line:** Treatment risks are much smaller than the risks of untreated sickle cell disease!"
        return response, False

    elif question_type == 'bone_marrow_danger':
        response += "**Is bone marrow transplant dangerous? What about for an elderly lady?**\n\n"
        response += "**Bone marrow transplant risks vary significantly by age:**\n\n"
        response += "**For younger patients (under 16):**\n"
        response += "• Success rate: 90-95%\n• Lower risk of complications\n• Faster recovery\n• Long-term outcomes excellent\n\n"
        response += "**For adults (16-40):**\n"
        response += "• Success rate: 75-85%\n• Moderate risk of complications\n• Longer recovery time\n• Still generally good outcomes\n\n"
        response += "**For elderly patients (over 50-60):**\n"
        response += "• ⚠️ **Much higher risks**\n• Success rates lower (50-70%)\n• Higher chance of complications\n• Slower recovery\n• May not be recommended depending on overall health\n\n"
        response += "**Specific risks for elderly:**\n"
        response += "• Higher risk of graft-versus-host disease\n• Greater chance of infections\n• Heart and lung complications more likely\n• Recovery takes much longer\n\n"
        response += "**Decision factors for elderly patients:**\n"
        response += "• Overall health status\n• Severity of sickle cell symptoms\n• Quality of life with current treatments\n• Family support system\n\n"
        response += "**Bottom line:** For elderly patients, doctors carefully weigh risks vs. benefits. Many may be better managed with medications rather than transplant."
        return response, False

    elif question_type == 'selfcare':
        response += "**Self-Care and Management for Sickle Cell Disease**\n\n"
        response += "Managing sickle cell disease involves daily habits, medical care, and lifestyle adjustments that reduce complications.\n\n"
        response += "**Key self-care tips include:**\n"
        response += "• **Stay hydrated:** Drink plenty of water to keep blood flowing smoothly\n"
        response += "• **Avoid extreme temperatures:** Cold or hot weather can trigger pain crises\n"
        response += "• **Eat a balanced diet:** Include leafy greens, fruits, and whole grains\n"
        response += "• **Get enough rest:** Fatigue is common — prioritize sleep and rest\n"
        response += "• **Prevent infections:** Wash hands, stay updated on vaccines, and avoid sick people\n"
        response += "• **Manage stress:** Use relaxation techniques like deep breathing or meditation\n"
        response += "• **Follow treatment plans:** Take medications (like hydroxyurea) as prescribed\n\n"
        response += "**Key message:**\n"
        response += "Good self-care reduces complications and improves quality of life. Regular doctor visits and healthy habits make a big difference."
        return response, False

    elif question_type == 'symptoms':
        response += "**Symptoms of Sickle Cell Disease**\n\n"
        response += "Sickle cell disease can cause a wide range of symptoms that vary from person to person and over time.\n\n"
        response += "**Most common symptoms include:**\n"
        response += "• Episodes of pain (called sickle cell crises)\n"
        response += "• Fatigue or low energy\n"
        response += "• Swelling in the hands and feet\n"
        response += "• Frequent infections\n"
        response += "• Delayed growth and puberty\n"
        response += "• Vision problems\n"
        response += "• Pale or yellowish skin (jaundice)\n\n"
        response += "**Pain crises:**\n"
        response += "• Sudden and severe pain, often in the chest, joints, or abdomen\n"
        response += "• Caused by blocked blood flow due to sickle-shaped cells\n\n"
        response += "**Key message:**\n"
        response += "Symptoms can be managed with medication, hydration, rest, and regular checkups. Always consult a doctor if symptoms worsen or change."
        return response, False

    elif question_type == 'blood_effect':
        response += "**🩸 How Does Sickle Cell Disease Affect Blood Cells?**\n\n"
        response += "Sickle cell disease causes red blood cells to become **abnormally shaped**, like a crescent or sickle, instead of the usual round and flexible shape.\n\n"
        response += "**These sickle-shaped cells:**\n"
        response += "• Are **rigid and sticky**\n"
        response += "• Can **clump together** and block blood flow\n"
        response += "• **Break down easily**, leading to low red blood cell counts (anemia)\n\n"
        response += "**Effects on the blood and body:**\n"
        response += "• **Anemia:** Due to shorter lifespan of sickle cells (10–20 days vs 120 days for normal cells)\n"
        response += "• **Pain crises:** Caused by blocked blood flow to organs and joints\n"
        response += "• **Organ damage:** From lack of oxygen due to poor blood circulation\n"
        response += "• **Fatigue & weakness:** Because of reduced oxygen delivery throughout the body\n\n"
        response += "**Key message:** Sickle cell disease primarily affects red blood cells, which leads to complications throughout the entire body due to poor oxygen delivery."
        return response, False

    elif question_type == 'trait_vs_disease':
        response += "**What's the difference between sickle cell trait and sickle cell disease?**\n\n"
        response += "**Sickle Cell Trait (SCT):**\n"
        response += "• Have ONE sickle gene + ONE normal gene\n• Usually NO symptoms at all\n• Live completely normal lives\n• Don't need any treatment\n• About 40% sickle hemoglobin, 60% normal\n• Can pass the gene to children\n\n"
        response += "**Sickle Cell Disease (SCD):**\n"
        response += "• Have TWO sickle genes (one from each parent)\n• Experience symptoms like pain crises\n• Need ongoing medical care\n• About 80-90% sickle hemoglobin\n• Will pass sickle gene to ALL children\n\n"
        response += "**Inheritance patterns:**\n"
        response += "• **Trait × Normal:** 50% chance trait, 50% normal children\n• **Trait × Trait:** 25% disease, 50% trait, 25% normal children\n• **Disease × Normal:** 100% trait children\n• **Disease × Trait:** 50% disease, 50% trait children\n• **Disease × Disease:** 100% disease children\n\n"
        response += "**Key difference:** Trait = carrier (usually healthy), Disease = affected (needs medical care)\n\n"
        response += "**Both are important to know about for family planning decisions!**"
        return response, False

    elif question_type == 'gender_inheritance':
        response += "**Can both boys and girls inherit sickle cell equally?**\n\n"
        response += "**Yes, absolutely! Sickle cell disease affects boys and girls equally.**\n\n"
        response += "**Why it's equal:**\n"
        response += "• The sickle cell gene is NOT on the X or Y chromosome\n• It's on chromosome 11, which both boys and girls have two copies of\n• Each parent passes one copy to each child, regardless of gender\n\n"
        response += "**Inheritance is the same:**\n"
        response += "• 50% chance for each child if both parents have trait\n• 25% chance if both parents have trait (for disease)\n• Gender doesn't change these percentages at all\n\n"
        response += "**However, some differences in experience:**\n"
        response += "• **Girls:** May have additional challenges during menstruation (blood loss can worsen anemia)\n• **Boys:** No significant gender-specific differences\n• **Pregnancy:** Women need extra monitoring during pregnancy\n• **Pain crises:** Affect both genders equally\n\n"
        response += "**Statistics worldwide:**\n"
        response += "• Roughly 50% male, 50% female patients\n• No gender is 'protected' from inheriting it\n\n"
        response += "**Bottom line:** Sickle cell doesn't discriminate by gender - it affects boys and girls equally!"
        return response, False

    elif question_type == 'early_signs':
        response += "**What are early signs of sickle cell anemia in young kids?**\n\n"
        response += "**Very early signs (6 months - 2 years):**\n\n"
        response += "**Hand-Foot Syndrome (often the FIRST sign):**\n"
        response += "• Swelling of hands and feet\n• Baby crying when hands/feet are touched\n• Hands/feet may feel warm\n• Usually happens around 6-18 months\n\n"
        response += "**Other early signs:**\n"
        response += "• **Excessive fussiness/crying** (due to pain)\n• **Frequent infections** (colds, pneumonia)\n• **Pale skin, lips, or nail beds** (anemia)\n• **Fatigue/tiredness** more than other babies\n• **Slow growth** compared to other children\n• **Yellow tinge to eyes or skin** (jaundice)\n\n"
        response += "**In toddlers (2-5 years):**\n"
        response += "• Complaining of pain in arms, legs, chest, or back\n• Not wanting to walk or play as much\n• Frequent fevers\n• Bedwetting (from kidney problems)\n• Delayed milestones\n\n"
        response += "**When to be concerned:**\n"
        response += "• Any combination of these symptoms\n• Family history of sickle cell\n• Symptoms that keep coming back\n\n"
        response += "**Important:** Many countries now test ALL newborns for sickle cell, so most cases are caught before symptoms appear. Early detection allows for preventive treatment!"
        return response, False

    elif question_type == 'untreated':
        response += "**What happens if sickle cell is left untreated or ignored?**\n\n"
        response += "**Serious complications can develop:**\n\n"
        response += "**Short-term consequences:**\n"
        response += "• More frequent and severe pain crises\n• Life-threatening infections (especially in young children)\n• Acute chest syndrome (lung complications)\n• Stroke (especially in children)\n• Severe anemia requiring blood transfusions\n\n"
        response += "**Long-term organ damage:**\n"
        response += "• **Heart:** Enlarged heart, heart failure\n• **Kidneys:** Kidney damage, kidney failure\n• **Lungs:** Chronic lung problems\n• **Eyes:** Vision problems, blindness\n• **Bones:** Bone damage, hip problems\n• **Brain:** Cognitive problems from small strokes\n• **Liver:** Liver damage\n\n"
        response += "**Other serious effects:**\n"
        response += "• Delayed growth and development\n• Delayed puberty\n• Leg ulcers that don't heal\n• Gallstones\n• Reduced life expectancy\n\n"
        response += "**Can it be cured if treated late?**\n"
        response += "• **Organ damage:** Usually permanent and cannot be reversed\n• **Future symptoms:** Can still be controlled with treatment\n• **Life expectancy:** Can be improved even with late treatment\n• **Quality of life:** Significantly better with treatment at any age\n\n"
        response += "**Key message:** It's never too late to start treatment, but early treatment prevents permanent damage!"
        return response, False

    elif question_type == 'pain_crisis':
        response += "**Why do people with sickle cell experience pain crises?**\n\n"
        response += "**The basic mechanism:**\n\n"
        response += "**What happens during a crisis:**\n"
        response += "• Sickle-shaped red blood cells become rigid and sticky\n• These abnormal cells clump together\n• They block small blood vessels (like a traffic jam)\n• Tissues and organs don't get enough oxygen and nutrients\n• This causes severe pain\n\n"
        response += "**Why the pain is so intense:**\n"
        response += "• It's similar to a heart attack, but in different parts of the body\n• When tissues don't get oxygen, they literally start to die\n• The pain signals this tissue damage\n• Multiple areas can be affected at once\n\n"
        response += "**Common triggers for pain crises:**\n"
        response += "• **Dehydration** (most common trigger)\n• Infections or fever\n• Extreme temperatures (hot or cold)\n• High altitude (less oxygen)\n• Physical or emotional stress\n• Certain medications\n• Sometimes no obvious trigger\n\n"
        response += "**Where pain typically occurs:**\n"
        response += "• Bones and joints (back, arms, legs, chest)\n• Abdomen\n• Chest\n• Can affect any part of the body\n\n"
        response += "**Duration and intensity:**\n"
        response += "• Can last hours to days\n• Pain can be mild to excruciating\n• Often described as the worst pain imaginable\n\n"
        response += "**Prevention is key:** Staying hydrated and avoiding triggers can reduce frequency of crises."
        return response, False

    elif question_type == 'organs_affected':
        response += "**What organs are affected by sickle cell disease?**\n\n"
        response += "**Sickle cell can affect virtually every organ system:**\n\n"
        response += "**Heart:**\n"
        response += "• Enlarged heart (working harder to pump blood)\n• Heart murmurs\n• Eventually heart failure if not managed\n\n"
        response += "**Lungs:**\n"
        response += "• Acute chest syndrome (life-threatening lung complication)\n• Chronic lung disease\n• Increased risk of pneumonia\n\n"
        response += "**Brain:**\n"
        response += "• Stroke (especially in children)\n• Silent strokes (small strokes causing learning problems)\n• Seizures\n• Cognitive difficulties\n\n"
        response += "**Kidneys:**\n"
        response += "• Kidney damage over time\n• Problems concentrating urine\n• Kidney failure (in severe cases)\n• Blood in urine\n\n"
        response += "**Eyes:**\n"
        response += "• Retinal damage\n• Vision problems\n• Potential blindness\n\n"
        response += "**Bones and Joints:**\n"
        response += "• Bone pain and damage\n• Hip problems (avascular necrosis)\n• Growth delays\n\n"
        response += "**Liver and Gallbladder:**\n"
        response += "• Liver damage\n• Gallstones (very common)\n• Jaundice\n\n"
        response += "**Spleen:**\n"
        response += "• Spleen damage or loss of function\n• Increased infection risk\n\n"
        response += "**Skin:**\n"
        response += "• Leg ulcers (especially in adults)\n• Slow healing wounds\n\n"
        response += "**The good news:** With proper medical care, most organ damage can be prevented or minimized!"
        return response, False

    elif question_type == 'growth_development':
        response += "**How does sickle cell affect growth, development, and fertility?**\n\n"
        response += "**Growth Effects:**\n"
        response += "• Children often grow slower than peers\n• May be shorter and weigh less\n• Growth spurts may be delayed\n• With good medical care, most catch up eventually\n• Proper nutrition and treatment help normal growth\n\n"
        response += "**Development Effects:**\n"
        response += "• **Physical development:** May be delayed but usually normal eventually\n• **Puberty:** Often delayed by 1-2 years\n• **Cognitive development:** Usually normal intelligence\n• **Learning:** Some children may have learning difficulties due to silent strokes\n• **Motor skills:** Generally develop normally\n\n"
        response += "**Fertility Effects:**\n\n"
        response += "**For Women:**\n"
        response += "• Most women with sickle cell can get pregnant\n• May have irregular menstrual periods\n• Pregnancy needs extra medical monitoring\n• Higher risk of complications during pregnancy\n• Can breastfeed normally\n\n"
        response += "**For Men:**\n"
        response += "• Most men have normal fertility\n• Some may experience priapism (painful erections)\n• Sperm count usually normal\n• Can father children normally\n\n"
        response += "**Pregnancy considerations:**\n"
        response += "• Genetic counseling recommended\n• 25% chance of sickle cell disease if partner also has trait\n• Prenatal testing available\n• Extra medical care needed during pregnancy\n\n"
        response += "**Key message:** Most people with sickle cell can have normal development and families with proper medical support!"
        return response, False

    elif question_type == 'prenatal':
        response += "**Can sickle cell be detected before birth?**\n\n"
        response += "**Yes! Sickle cell disease can be detected during pregnancy.**\n\n"
        response += "**Prenatal testing options:**\n\n"
        response += "**Chorionic Villus Sampling (CVS):**\n"
        response += "• Done at 10-13 weeks of pregnancy\n• Small sample taken from placenta\n• 99% accurate\n• Small risk of miscarriage (less than 1 in 300)\n\n"
        response += "**Amniocentesis:**\n"
        response += "• Done at 15-20 weeks of pregnancy\n• Sample of amniotic fluid taken\n• 99% accurate\n• Small risk of miscarriage (less than 1 in 500)\n\n"
        response += "**Who should consider testing:**\n"
        response += "• Both parents have sickle cell trait\n• One parent has sickle cell disease\n• Family history of sickle cell disease\n• Parents are from high-risk ethnic groups\n\n"
        response += "**Newer options:**\n"
        response += "• **Non-invasive prenatal testing (NIPT):** Blood test from mother, no risk to baby\n• Still being developed for sickle cell\n• May be available in some centers\n\n"
        response += "**What the results mean:**\n"
        response += "• Testing can tell if baby will have disease, trait, or be normal\n• Helps parents prepare for medical care if needed\n• Allows for early treatment planning\n\n"
        response += "**Important:** Prenatal testing is a personal choice. Genetic counseling can help parents understand options and make informed decisions."
        return response, False

    elif question_type == 'donor_eligibility':
        response += "**Who is eligible to give bone marrow for sickle cell treatment?**\n\n"
        response += "**Best donors (in order of preference):**\n\n"
        response += "**1. Siblings (Brothers/Sisters):**\n"
        response += "• **Best option** - about 25% chance of being a perfect match\n• Must have compatible tissue type (HLA matching)\n• Should NOT have sickle cell disease themselves\n• Can have sickle cell trait (that's actually okay)\n\n"
        response += "**2. Parents:**\n"
        response += "• Usually only half-matches\n• Sometimes used, but with higher risk\n• Results not as good as sibling matches\n\n"
        response += "**3. Other Family Members:**\n"
        response += "• Cousins, aunts, uncles - rarely good matches\n• Very low chance of compatibility\n\n"
        response += "**4. Unrelated Donors:**\n"
        response += "• From bone marrow donor registries\n• Much harder to find good matches\n• Especially difficult for people of African, Mediterranean, or Middle Eastern ancestry\n• Success rates lower than family donors\n\n"
        response += "**Requirements for donors:**\n"
        response += "• Generally healthy\n• Age 18-55 (for unrelated donors)\n• No serious medical conditions\n• Compatible blood type helps but isn't essential\n• Willing to go through donation process\n\n"
        response += "**Testing process:**\n"
        response += "• Blood test for HLA typing\n• Medical evaluation\n• Psychological evaluation\n\n"
        response += "**Reality:** Only about 20-30% of patients have a suitable donor, which is why other treatments are important too."
        return response, False

    elif question_type == 'treatment_risks':
        response += "**⚠️ What Are the Risks or Side Effects of Sickle Cell Treatments?**\n\n"

        response += "**💊 Hydroxyurea (Most common medication):**\n"
        response += "• Lower white blood cell count (temporary)\n"
        response += "• Nausea, loss of appetite\n"
        response += "• Skin or nail darkening\n"
        response += "• Rare: Fertility effects, blood cancers (very rare)\n\n"

        response += "**💉 Blood Transfusions:**\n"
        response += "• Iron overload (may need chelation)\n"
        response += "• Very low risk of infections\n"
        response += "• Rare allergic reactions or antibodies\n\n"

        response += "**🏥 Bone Marrow Transplant (Curative but high-risk):**\n"
        response += "• 5-10% risk of serious complications\n"
        response += "• Rejection (Graft-vs-Host Disease)\n"
        response += "• Infection during recovery\n"
        response += "• Fertility loss, organ damage (in rare cases)\n\n"

        response += "**🧬 Gene Therapy (Experimental):**\n"
        response += "• Long-term risks not fully known\n"
        response += "• Expensive\n"
        response += "• Early results are promising\n\n"

        response += "**💊 Pain Medications (Opioids, NSAIDs):**\n"
        response += "• Constipation, drowsiness, liver stress\n"
        response += "• Risk of dependence if overused\n\n"

        response += "**✅ Final Takeaway:**\n"
        response += "• Most treatments are safe when monitored properly\n"
        response += "• Doctors weigh risks vs. benefits before prescribing\n"
        response += "• Always follow up regularly and report side effects early\n"
        response += "• Never stop treatment without consulting a hematologist"
        return response, False
    
    elif question_type == 'treatment_types':
        response += "**What are permanent vs temporary treatments? How effective are they?**\n\n"
        response += "**TEMPORARY TREATMENTS (Managing the disease):**\n\n"
        response += "**Medications:**\n"
        response += "• **Hydroxyurea:** Reduces pain crises by 50-70%\n• **Pain medications:** Control pain during crises\n• **Antibiotics:** Prevent infections\n• **Blood transfusions:** For severe cases\n• **Folic acid:** Helps make new red blood cells\n\n"
        response += "**Effectiveness:** 70-90% of patients see significant improvement in symptoms\n\n"
        response += "**PERMANENT TREATMENTS (Potential cures):**\n\n"
        response += "**Bone Marrow Transplant:**\n"
        response += "• Success rate: 85-95% when good donor available\n• Best for children under 16\n• Requires compatible donor (usually sibling)\n• Can completely cure the disease\n\n"
        response += "**Gene Therapy (Experimental):**\n"
        response += "• Early trials show 80-90% success\n• Still in clinical trials\n• May become widely available in 5-10 years\n\n"
        response += "**Bottom line:** Temporary treatments work very well for most people, permanent cures exist but aren't suitable for everyone."
        return response, False

    elif question_type == 'lifestyle':
        response += "**How Should Someone with Sickle Cell Adapt Their Lifestyle?**\n\n"
        response += "**MOST IMPORTANT - Stay Hydrated:**\n"
        response += "• Drink 8-10 glasses of water daily\n• Dehydration is the #1 trigger for pain crisis\n• Carry water bottle everywhere\n\n"
        response += "**Avoid Temperature Extremes:**\n"
        response += "• Don't get too hot or too cold\n• Use air conditioning in summer\n• Dress warmly in winter\n• Avoid ice baths or very hot showers\n\n"
        response += "**Get Enough Rest:**\n"
        response += "• Sleep 7-8 hours nightly\n• Avoid excessive physical stress\n• Take breaks during activities\n\n"
        response += "**Diet Changes:**\n"
        response += "• Eat iron-rich foods (spinach, beans)\n• Take folic acid supplements\n• Avoid alcohol (can trigger crisis)\n• Eat regular, balanced meals\n\n"
        response += "**Exercise Wisely:**\n"
        response += "• Light to moderate exercise is good\n• Avoid intense, exhausting workouts\n• Swimming is excellent (if water isn't too cold)\n• Stop if you feel tired\n\n"
        response += "**These changes can reduce pain crises by 40-60%!**"
        return response, False

    elif question_type == 'treatment_locations':
        response += "**🏥 Best Hospitals and Centers for Sickle Cell Treatment**\n\n"

        response += "**Top Hospitals Worldwide:**\n"
        response += "• **St. Jude Children’s Research Hospital (USA)** – Known for pediatric sickle cell care and clinical trials\n"
        response += "• **NIH Clinical Center (USA)** – Offers gene therapy trials\n"
        response += "• **King’s College Hospital (UK)** – Home to Europe’s largest sickle cell unit\n"
        response += "• **Apollo Hospitals (India)** – Offers bone marrow transplants and hematology\n"
        response += "• **SickKids Hospital (Canada)** – Excellent for pediatric sickle cell management\n"
        response += "• **INSERM/Necker Hospital (France)** – Advanced care and research\n\n"

        response += "**What Makes a Center Great:**\n"
        response += "• Specialized hematologists\n"
        response += "• Bone marrow transplant programs\n"
        response += "• Access to clinical trials\n"
        response += "• Genetic counseling and long-term care\n\n"

        response += "**Tip:** Large academic hospitals or government-approved sickle cell centers often offer the best outcomes."
        return response, False
        
    elif question_type == 'sickle_types':
        response += "**What types of sickle cell disease are there?**\n\n"
        response += "**Main types (from most to least severe):**\n\n"
        response += "**1. HbSS (Sickle Cell Anemia) - Most Severe**\n"
        response += "• Both parents passed sickle cell gene\n• Most painful crises\n• Needs most medical care\n• About 65% of all cases\n\n"
        response += "**2. HbSC Disease - Moderate**\n"
        response += "• One sickle gene + one C gene\n• Milder than HbSS\n• Still needs medical care\n• About 25% of cases\n\n"
        response += "**3. HbS Beta-Thalassemia - Variable**\n"
        response += "• Sickle gene + thalassemia gene\n• Can be mild or severe\n• Two subtypes: Beta+ (milder) and Beta0 (severe)\n\n"
        response += "**4. HbAS (Sickle Cell Trait) - Usually Harmless**\n"
        response += "• Only one sickle gene\n• Usually no symptoms\n• Can pass gene to children\n• About 8% of African Americans have this\n\n"
        response += "**Rare types:** HbSD, HbSE, HbSO - very uncommon\n\n"
        response += "**How to know which type:** Need blood test called hemoglobin electrophoresis."
        return response, False

    elif question_type == 'type_detection':
        response += "**Can you detect the type of sickle cell from an image?**\n\n"
        response += "**Short answer: NO** - Images alone cannot determine the specific type.\n\n"
        response += "**What images CAN show:**\n"
        response += "• Whether sickle cells are present or not\n• Severity of sickling\n• General shape abnormalities\n\n"
        response += "**What images CANNOT show:**\n"
        response += "• Specific type (HbSS, HbSC, etc.)\n• Exact genetic makeup\n• Hemoglobin composition\n\n"
        response += "**To confirm the type, you need these lab tests:**\n\n"
        response += "**1. Hemoglobin Electrophoresis** - Most important\n"
        response += "• Shows exact type of hemoglobin\n• Distinguishes HbSS from HbSC, etc.\n• Gold standard test\n\n"
        response += "**2. HPLC (High Performance Liquid Chromatography)**\n"
        response += "• More precise than electrophoresis\n• Quantifies different hemoglobin types\n\n"
        response += "**3. DNA Analysis/Genetic Testing**\n"
        response += "• Shows exact genetic mutations\n• Most accurate but expensive\n\n"
        response += "**4. Solubility Test (Sickledex)**\n"
        response += "• Quick screening test\n• Only shows if sickle hemoglobin is present\n\n"
        if is_sickle:
            response += f"\n**For your image:** I can see sickle cells ({confidence*100:.1f}% confidence), but you'll need the lab tests above to know the exact type."
        return response, False

    elif question_type == 'blood_smear':
        response += "**What can a blood smear tell about sickle cell?**\n\n"
        response += "**A blood smear can reveal:**\n\n"
        response += "**Cell Shape:**\n"
        response += "• Sickle-shaped (crescent) cells\n• Elongated, rigid cells\n• Target cells (cells with bull's-eye appearance)\n• Howell-Jolly bodies (small dots in cells)\n\n"
        response += "**Cell Count & Size:**\n"
        response += "• Low red blood cell count (anemia)\n• Larger than normal red cells\n• Immature red cells (reticulocytes)\n\n"
        response += "**Signs of Complications:**\n"
        response += "• Fragmented cells (from blocked blood vessels)\n• White blood cell changes\n• Platelet count changes\n\n"
        response += "**What it CANNOT tell:**\n"
        response += "• Exact type of sickle cell disease\n• Severity of symptoms\n• How well treatments will work\n• Carrier status definitively\n\n"
        response += "**Limitations:**\n"
        response += "• Cells may look normal between crises\n• Some people with trait show no sickle cells\n• Need special preparation to see sickling\n\n"
        response += "**Bottom line:** Blood smear is helpful for diagnosis but needs to be combined with other tests for complete picture."
        if is_sickle:
            response += f"\n\n**Your blood smear:** Shows signs consistent with sickle cell disease ({confidence*100:.1f}% confidence)."
        return response, False

    elif question_type == 'checkup_frequency':
        response += "**How often do I need checkups and tests?**\n\n"
        response += "**For Adults with Sickle Cell Disease:**\n\n"
        response += "**Every 3-6 months (Regular checkups):**\n"
        response += "• Complete blood count (CBC)\n• Liver and kidney function tests\n• Blood pressure check\n• Weight and general health\n\n"
        response += "**Every 6-12 months:**\n"
        response += "• Eye exam (retinal screening)\n• Lung function tests\n• Heart function (ECG/Echo)\n• Bone density scan\n\n"
        response += "**Yearly:**\n"
        response += "• Transcranial Doppler (stroke screening)\n• Comprehensive metabolic panel\n• Immunizations update\n• Pulmonary hypertension screening\n\n"
        response += "**For Children - More Frequent:**\n"
        response += "• Every 2-3 months for routine care\n• Growth and development monitoring\n• More frequent eye and brain scans\n\n"
        response += "**Emergency visits when:**\n"
        response += "• Fever over 101.3°F (38.5°C)\n• Severe pain that doesn't respond to home treatment\n• Difficulty breathing\n• Severe headache or vision changes\n• Signs of stroke\n\n"
        response += "**Special situations need more frequent visits:**\n"
        response += "• Pregnancy\n• Recent complications\n• Starting new medications\n• Before/after surgery"
        if is_sickle:
            response += f"\n\n**Since your blood test suggests sickle cell disease ({confidence*100:.1f}% confidence), please establish care with a hematologist soon.**"
        return response, False

    elif question_type == 'ayurvedic':
        response += "**Is there any Ayurvedic treatment for sickle cell?**\n\n"
        response += "**Ayurvedic approaches being studied:**\n\n"
        response += "**Herbal remedies with some research:**\n"
        response += "• **Cajanus cajan (Pigeon pea)** - May reduce sickling\n• **Fagara zanthoxyloides** - Anti-sickling properties\n• **Terminalia catappa** - Antioxidant effects\n• **Carica papaya** - May help with pain\n\n"
        response += "**Traditional Ayurvedic treatments:**\n"
        response += "• Panchakarma detoxification\n• Rasayana therapy (rejuvenation)\n• Specific dietary recommendations\n• Yoga and meditation for pain management\n\n"
        response += "**What research shows:**\n"
        response += "• Some herbs may reduce pain and sickling\n• Anti-inflammatory effects documented\n• May help with overall well-being\n• Limited large-scale clinical trials\n\n"
        response += "**⚠️ IMPORTANT WARNINGS:**\n"
        response += "• **NEVER replace modern medicine with Ayurveda alone**\n• Use only as complementary therapy\n• Always inform your hematologist about any herbal medicines\n• Some herbs can interact with medications\n• Quality and purity of herbal products varies\n\n"
        response += "**Best approach:**\n"
        response += "• Continue standard medical treatment\n• Add Ayurvedic therapies under supervision\n• Find qualified Ayurvedic practitioners\n• Regular monitoring by both doctors\n\n"
        response += "**Bottom line:** Ayurveda can be helpful as additional support, but modern medicine remains essential for sickle cell disease."
        return response, False

    elif question_type == 'home_remedy':
        response += "**What should I do during pain crisis at home?**\n\n"
        response += "**Immediate home management:**\n\n"
        response += "**1. HYDRATE HEAVILY**\n"
        response += "• Drink water every 15-20 minutes\n• Warm fluids are better than cold\n• Avoid alcohol and caffeine\n\n"
        response += "**2. HEAT THERAPY**\n"
        response += "• Warm bath or shower\n• Heating pads on painful areas\n• Warm compress (not too hot)\n• Avoid ice or cold packs\n\n"
        response += "**3. PAIN RELIEF**\n"
        response += "• Take prescribed pain medications as directed\n• Ibuprofen or acetaminophen for mild pain\n• Don't wait for pain to get worse\n\n"
        response += "**4. REST AND POSITIONING**\n"
        response += "• Lie down in comfortable position\n• Elevate painful limbs\n• Gentle stretching if tolerable\n• Avoid strenuous activity\n\n"
        response += "**5. BREATHING EXERCISES**\n"
        response += "• Deep, slow breathing\n• Meditation or relaxation techniques\n• Helps manage pain and anxiety\n\n"
        response += "**6. AVOID TRIGGERS**\n"
        response += "• Stay warm\n• Avoid stress\n• Don't smoke\n• Avoid dehydration\n\n"
        response += "**⚠️ GO TO HOSPITAL IMMEDIATELY IF:**\n"
        response += "• Fever over 101.3°F (38.5°C)\n• Difficulty breathing\n• Severe chest pain\n• Severe headache\n• Vision changes\n• Weakness or numbness\n• Pain not responding to home treatment after 2-3 hours\n• Vomiting and can't keep fluids down\n\n"
        response += "**🚨 Don't delay hospital visit - go as soon as possible if you have any warning signs! Sickle cell crises can become life-threatening quickly.**"
        return response, False

    elif question_type == 'hospital_timing':
        response += "**When should I go to the hospital?**\n\n"
        response += "**🚨 GO IMMEDIATELY (Call 911 or Emergency):**\n\n"
        response += "**Fever:**\n"
        response += "• Temperature 101.3°F (38.5°C) or higher\n• Even if you feel okay otherwise\n\n"
        response += "**Breathing problems:**\n"
        response += "• Shortness of breath\n• Chest pain\n• Fast breathing\n• Coughing up blood\n\n"
        response += "**Neurological signs:**\n"
        response += "• Severe headache\n• Vision changes\n• Weakness on one side\n• Confusion\n• Seizures\n• Trouble speaking\n\n"
        response += "**Severe pain:**\n"
        response += "• Pain not relieved by home treatment after 2-3 hours\n• Pain getting worse despite medication\n• Can't function or sleep due to pain\n\n"
        response += "**Other emergency signs:**\n"
        response += "• Yellowing of eyes/skin (jaundice)\n• Severe fatigue/weakness\n• Painful erection lasting >4 hours\n• Severe abdominal pain\n• Can't keep fluids down\n\n"
        response += "**⏰ GO WITHIN FEW HOURS:**\n"
        response += "• Moderate pain not improving\n• Swelling in hands/feet\n• Leg ulcers getting worse\n• Signs of infection\n\n"
        response += "**📞 CALL YOUR DOCTOR FIRST:**\n"
        response += "• Mild pain crisis\n• Questions about medications\n• Routine concerns\n\n"
        response += "**🏥 REMEMBER:**\n"
        response += "• Don't wait and see if it gets better\n• It's better to go early than too late\n• Emergency rooms understand sickle cell emergencies\n• Bring your medication list and medical records if possible\n\n"
        response += "**⚠️ NEVER ignore fever or breathing problems - these can be life-threatening in sickle cell patients!**"
        return response, False

    elif question_type == 'travel':
        response += "**Can I travel? What precautions should I take?**\n\n"
        response += "**Yes, you can travel! But with precautions:**\n\n"
        response += "**Before traveling:**\n\n"
        response += "**Medical preparation:**\n"
        response += "• Get doctor's clearance\n• Get written medical summary\n• Ensure vaccinations are up to date\n• Get travel insurance that covers pre-existing conditions\n\n"
        response += "**Pack medications:**\n"
        response += "• Extra supply (2x what you need)\n• Keep in carry-on luggage\n• Bring prescription letters\n• Include pain medications and antibiotics\n\n"
        response += "**During travel:**\n\n"
        response += "**Air travel tips:**\n"
        response += "• Request aisle seat for easy movement\n• Walk every 1-2 hours\n• Stay hydrated (drink water frequently)\n• Avoid alcohol\n• Ask for oxygen if feeling unwell\n\n"
        response += "**At destination:**\n\n"
        response += "**Climate considerations:**\n"
        response += "• Avoid extreme temperatures\n• Don't swim in very cold water\n• Stay in air-conditioned accommodations\n• Dress appropriately for weather\n\n"
        response += "**High altitude precautions:**\n"
        response += "• Avoid places above 10,000 feet\n• Ascend slowly if going to moderate altitude\n• Watch for breathing problems\n• Consider oxygen supplementation\n\n"
        response += "**General travel tips:**\n"
        response += "• Find nearest hospital at destination\n• Carry emergency contact information\n• Maintain regular sleep schedule\n• Eat regularly and stay hydrated\n• Avoid excessive physical exertion\n\n"
        response += "**Places to be extra careful:**\n"
        response += "• High altitude destinations\n• Very cold climates\n• Areas with limited medical facilities\n• Places with disease outbreaks\n\n"
        response += "**Emergency plan:**\n"
        response += "• Know how to access healthcare abroad\n• Have emergency contacts readily available\n• Know your blood type and medication allergies\n• Consider medical evacuation insurance"
        return response, False

    elif question_type == 'safe_childbearing':
        response += "**If I have sickle cell, how can I have children safely?**\n\n"
        response += "**Yes, you can have children safely with proper planning!**\n\n"
        response += "**Before pregnancy (Pre-conception planning):**\n\n"
        response += "**Genetic counseling:**\n"
        response += "• Test your partner for sickle cell trait\n• Understand risks to baby\n• If both parents have trait/disease: 25% chance baby has disease\n• Consider all options\n\n"
        response += "**Health optimization:**\n"
        response += "• Start folic acid supplements (5mg daily)\n• Ensure vaccinations are current\n• Control pain and optimize treatment\n• Achieve good nutritional status\n\n"
        response += "**During pregnancy:**\n\n"
        response += "**Specialized care needed:**\n"
        response += "• High-risk pregnancy specialist\n• Hematologist involvement\n• More frequent check-ups\n• Extra monitoring of baby\n\n"
        response += "**Common pregnancy complications:**\n"
        response += "• More frequent pain crises\n• Higher risk of infections\n• Preeclampsia\n• Preterm labor\n• Growth restriction in baby\n\n"
        response += "**Special monitoring:**\n"
        response += "• Regular blood transfusions may be needed\n• Ultrasounds to check baby's growth\n• Monitoring for pregnancy complications\n• Hospital delivery recommended\n\n"
        response += "**Medication adjustments:**\n"
        response += "• Stop hydroxyurea (can harm baby)\n• Safe pain medications during pregnancy\n• Antibiotics for infection prevention\n\n"
        response += "**For the baby:**\n"
        response += "• Newborn screening for sickle cell\n• Early pediatric hematologist if needed\n• Special vaccinations if baby has disease\n\n"
        response += "**Success rates:**\n"
        response += "• With proper care, most pregnancies are successful\n• Modern medical care has greatly improved outcomes\n• Most babies are born healthy\n\n"
        response += "**Key message:** Plan ahead, get specialized care, and most women with sickle cell can have healthy pregnancies and babies!"
        return response, False

    elif question_type == 'doctor_specialist':
        response += "**Which doctor should I go to? What is the specialist called?**\n\n"
        response += "**Primary specialist: HEMATOLOGIST**\n\n"
        response += "**What is a hematologist?**\n"
        response += "• Doctor who specializes in blood disorders\n• Expert in sickle cell disease management\n• Can prescribe all sickle cell medications\n• Manages complications\n\n"
        response += "**Other doctors you may need:**\n\n"
        response += "**Pediatric Hematologist:**\n"
        response += "• For children with sickle cell disease\n• Specialized in childhood blood disorders\n\n"
        response += "**Pain Management Specialist:**\n"
        response += "• For chronic pain control\n• Expert in pain medications\n• Alternative pain treatments\n\n"
        response += "**Other specialists for complications:**\n"
        response += "• **Pulmonologist** - lung problems\n• **Cardiologist** - heart complications\n• **Nephrologist** - kidney problems\n• **Ophthalmologist** - eye complications\n• **Orthopedist** - bone problems\n• **Neurologist** - stroke prevention/management\n\n"
        response += "**How to find a good hematologist:**\n"
        response += "• Ask your primary care doctor for referral\n• Look for doctors at major hospitals\n• Check if they have sickle cell disease experience\n• Academic medical centers often have best specialists\n• Ask local sickle cell organizations\n\n"
        response += "**What to look for:**\n"
        response += "• Board certification in hematology\n• Experience with sickle cell patients\n• Affiliated with good hospital\n• Part of comprehensive sickle cell center\n\n"
        response += "**Red flags to avoid:**\n"
        response += "• Doctors who don't understand sickle cell\n• Those who dismiss your pain\n• Lack of experience with the disease\n• No access to emergency care\n\n"
        if is_sickle:
            response += f"\n**Since your blood test suggests sickle cell disease ({confidence*100:.1f}% confidence), I strongly recommend seeing a hematologist as soon as possible for proper diagnosis and treatment planning.**"
        return response, False

    elif question_type == 'treatment_locations':
        response += "**Which are the best places for sickle cell treatment?**\n\n"
        response += "**Best countries for treatment:**\n\n"
        response += "**1. United States**\n"
        response += "• Most advanced treatments available\n• Leading research centers\n• Best centers: Johns Hopkins, Duke, CHOP, Boston Children's\n• Gene therapy trials available\n\n"
        response += "**2. United Kingdom**\n"
        response += "• Excellent NHS sickle cell services\n• Good research programs\n• Free treatment for residents\n\n"
        response += "**3. France**\n"
        response += "• Strong sickle cell programs\n• Good outcomes\n• Research active\n\n"
        response += "**4. Canada**\n"
        response += "• Good universal healthcare coverage\n• Quality treatment centers\n\n"
        response += "**5. Germany**\n"
        response += "• Advanced medical care\n• Good bone marrow transplant programs\n\n"
        response += "**Top treatment centers worldwide:**\n\n"
        response += "**United States:**\n"
        response += "• Johns Hopkins (Baltimore)\n• Duke University (North Carolina)\n• Children's Hospital of Philadelphia\n• Boston Children's Hospital\n• St. Jude Children's Research Hospital\n\n"
        response += "**International:**\n"
        response += "• Great Ormond Street Hospital (London)\n• Hospital Necker (Paris)\n• King's College Hospital (London)\n• McMaster University (Canada)\n\n"
        response += "**What makes a center 'best':**\n"
        response += "• Comprehensive sickle cell programs\n• Research and clinical trials\n• Multiple specialists in one place\n• 24/7 emergency care\n• Bone marrow transplant capability\n• Social support services\n\n"
        response += "**For developing countries:**\n"
        response += "• Nigeria: University College Hospital Ibadan\n• Ghana: Korle Bu Teaching Hospital\n• India: AIIMS Delhi, CMC Vellore\n• Brazil: HEMORIO Rio de Janeiro\n\n"
        response += "**Cost considerations:**\n"
        response += "• US has best treatments but very expensive\n• European countries offer good care with lower costs\n• Some countries have medical tourism programs\n• Insurance coverage varies greatly"
        return response, False

    else:  # general questions
        if is_sickle:
            response += f"I've analyzed your blood sample and found evidence of sickle cell disease with {confidence*100:.1f}% confidence. I can see abnormal sickle-shaped cells in the blood smear."
            response += "I'd be happy to answer more specific questions about sickle cell disease, its symptoms, treatments, or anything else you'd like to know. You can ask me to highlight the sickle cells, explain symptoms, discuss treatments, or any other questions you might have.\n\n"
        else:
            response += f"Your blood sample looks normal and healthy! My analysis shows {confidence*100:.1f}% confidence that these are normal red blood cells without any signs of sickling.\n\n"
            response += "Feel free to ask me any questions about sickle cell disease in general, or if you have other concerns about blood health. I'm here to help explain anything you'd like to understand better."
    
    return response, False  # False means don't show processed image

# Enhanced VQA logic
def vqa_answer(question, image_path, prediction_label, confidence):
    if not question.strip():
        return "Please ask me a question about the blood sample or sickle cell disease!", None
    
    question_type, corrected_question, corrections_made = understand_question(question)
    
    if question_type == 'highlight':
        # For highlighting requests, always show the processed image
        if prediction_label == "Sickle Cell":
            result_img = highlight_cells(image_path)
        else:
            # For normal cells, just return the original image with a message
            img = cv2.imread(image_path)
            result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        response, _ = generate_response(...)
        return response, result_img
    else:
        response, open_tool = generate_response(...)
        return response, None, open_tool

# ----------------- Streamlit App ------------------

st.title("🧪 Sickle Cell Detector & Visual QA")
st.write("Upload a blood smear image and ask me questions about it!")

uploaded = st.file_uploader("📤 Upload a blood smear image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Get prediction
    label, confidence = classify_image(img)
    
    # Display prediction with appropriate styling
    if label == "Sickle Cell":
        st.error(f"🔍 **Analysis Result:** {label} detected with {confidence*100:.2f}% confidence")
        st.warning("⚠️ This suggests the presence of sickle cell disease. Please consult with a healthcare provider.")
    else:
        st.success(f"✅ **Analysis Result:** {label} with {confidence*100:.2f}% confidence")
        st.info("🎉 The blood sample appears normal and healthy!")

    # Save temp image for processing
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    image_path = os.path.join(temp_dir, "temp.jpg")
    img.save(image_path)

    st.markdown("---")
    st.subheader("💬 Ask Me Anything!")
    st.write("You can ask questions like:")
    st.write("• *Highlight sickle cells* | *Show abnormal cells*")
    st.write("• *What are the symptoms of sickle cell disease?*")
    st.write("• *Is this blood sample normal or abnormal?*")
    st.write("• *What causes sickle cell disease?*")
    st.write("• *How is sickle cell disease treated?*")

    question = st.text_input("🤔 Type your question here:", placeholder="e.g., What are the symptoms of sickle cell disease?")

    if question:
        with st.spinner("🧠 Analyzing your question..."):
        # Understand question
         q_type, corrected_q, corrections = understand_question(question)
        
        # Generate response
         answer, open_tool = generate_response(q_type, label, confidence, corrected_q, corrections)

        # Show image only for 'highlight' type
         if q_type == "highlight":
            highlighted_img = highlight_cells(image_path)
            st.markdown("### 🔍 Processed Image:")
            st.image(highlighted_img, caption="Highlighted Sickle Cells", use_column_width=True)
            st.markdown("### 💡 Answer:")
            st.write("🔴 Red boxes indicate confirmed sickle cells.")
            st.write("🟠 Orange boxes indicate doubtful cells that need human confirmation.")


         else:
            if open_tool:
                open_color_analysis_tool()
                st.success("✅ Color analysis tool opened in your browser!")
            else:
                st.markdown("### 💡 Answer:")
                st.markdown(answer)



    # Add helpful information section
    with st.expander("ℹ️ About This Tool"):
        st.write("""
        **What this tool does:**
        - Analyzes blood smear images for sickle cell disease
        - Highlights abnormal sickle-shaped cells
        - Answers questions about sickle cell disease naturally
        
        **Accuracy Note:**
        This is an AI analysis tool and should not replace professional medical diagnosis. 
        Always consult with healthcare providers for proper medical evaluation.
        
        **How to use:**
        1. Upload a blood smear image
        2. View the AI analysis result
        3. Ask any questions you have about the results or sickle cell disease in general
        """)

else:
    st.info("👆 Please upload a blood smear image to get started!")
    
    # Show sample questions when no image is uploaded
    st.markdown("### 🔬 Sample Questions You Can Ask:")
    sample_questions = [
        "What is sickle cell disease?",
        "What are the symptoms of sickle cell anemia?",
        "How is sickle cell disease inherited?",
        "What treatments are available for sickle cell disease?",
        "Is sickle cell disease dangerous?",
        "What foods should sickle cell patients avoid?",
        "How do sickle cells look different from normal cells?"
    ]
    
    for i, q in enumerate(sample_questions, 1):
        st.write(f"{i}. *{q}*")
