import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import sys
import importlib.util

# ===================================================
# CONFIGURATION
# ===================================================
device = tf.device('/CPU:0')

# ===================================================
# ENHANCED DUAL MODEL ANALYZER WITH CONNECTED VQA FILES
# ===================================================

class DualModelAnalyzer:
    def __init__(self):
        self.sickle_model = None
        self.iron_model = None
        self.sickle_vqa_module = None
        self.iron_vqa_module = None
        self.load_models()
        self.load_vqa_modules()

    def load_models(self):
        """Load both existing .h5 models"""
        print("Loading existing models...")
        
        # Load sickle cell model
        self.sickle_model = tf.keras.models.load_model(
            r"C:\Users\17bin\OneDrive\Desktop\Documents\PERSONAL_GROWTH\cHEAL\17july\merge\sickle_cell\model_fold5.h5"
        )
        print("✅ Loaded sickle cell model")
        
        # Load iron deficiency model  
        self.iron_model = tf.keras.models.load_model(
            r"C:\Users\17bin\OneDrive\Desktop\Documents\PERSONAL_GROWTH\cHEAL\17july\merge\iron_deficiency\final_dense_model.h5"
        )
        print("✅ Loaded iron deficiency model")

    def load_vqa_modules(self):
        """Load both VQA modules from existing files"""
        try:
            # Load sickle cell VQA (try1.py)
            sickle_vqa_path = r"C:\Users\17bin\OneDrive\Desktop\Documents\PERSONAL_GROWTH\cHEAL\17july\merge\sickle_cell\try1.py"
            if os.path.exists(sickle_vqa_path):
                spec = importlib.util.spec_from_file_location("sickle_vqa", sickle_vqa_path)
                self.sickle_vqa_module = importlib.util.module_from_spec(spec)
                sys.modules["sickle_vqa"] = self.sickle_vqa_module
                spec.loader.exec_module(self.sickle_vqa_module)
                print("✅ Loaded sickle cell VQA (try1.py)")
                
                # Check if vqa_answer function exists
                if hasattr(self.sickle_vqa_module, 'vqa_answer'):
                    print("✅ Found vqa_answer function in try1.py")
                else:
                    print("⚠ vqa_answer function not found in try1.py")
                    available = [f for f in dir(self.sickle_vqa_module) if not f.startswith('_') and callable(getattr(self.sickle_vqa_module, f))]
                    print(f"Available functions: {available}")
            else:
                print("⚠ Sickle VQA file not found at:", sickle_vqa_path)
                self.sickle_vqa_module = None
            
            # Load iron deficiency VQA (try2.py)  
            iron_vqa_path = r"C:\Users\17bin\OneDrive\Desktop\Documents\PERSONAL_GROWTH\cHEAL\17july\merge\iron_deficiency\try2.py"
            if os.path.exists(iron_vqa_path):
                spec = importlib.util.spec_from_file_location("iron_vqa", iron_vqa_path)
                self.iron_vqa_module = importlib.util.module_from_spec(spec)
                sys.modules["iron_vqa"] = self.iron_vqa_module
                spec.loader.exec_module(self.iron_vqa_module)
                print("✅ Loaded iron deficiency VQA (try2.py)")
                
                # Check if vqa_answer function exists
                if hasattr(self.iron_vqa_module, 'vqa_answer'):
                    print("✅ Found vqa_answer function in try2.py")
                else:
                    print("⚠ vqa_answer function not found in try2.py")
                    available = [f for f in dir(self.iron_vqa_module) if not f.startswith('_') and callable(getattr(self.iron_vqa_module, f))]
                    print(f"Available functions: {available}")
            else:
                print("⚠ Iron VQA file not found at:", iron_vqa_path)
                self.iron_vqa_module = None
                
        except Exception as e:
            print(f"Error loading VQA modules: {e}")
            import traceback
            traceback.print_exc()
            self.sickle_vqa_module = None
            self.iron_vqa_module = None

    def preprocess_for_sickle(self, image_path):
        """Preprocess image for sickle cell model (224x224)"""
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def preprocess_for_iron(self, image_path):
        """Preprocess image for iron deficiency model (160x160)"""
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array[..., ::-1]  # RGB -> BGR
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array

    def analyze_with_sickle_model(self, image_path):
        """Get prediction from sickle cell model"""
        image_array = self.preprocess_for_sickle(image_path)
        predictions = self.sickle_model.predict(image_array, verbose=0)
        
        sickle_prob = float(predictions[0][0])
        normal_prob = 1 - sickle_prob
        
        return {
            'sickle_probability': sickle_prob,
            'normal_probability': normal_prob,
            'prediction': 'sickle_cell' if sickle_prob > 0.5 else 'normal'
        }

    def analyze_with_iron_model(self, image_path):
        """Get prediction from iron deficiency model"""
        image_array = self.preprocess_for_iron(image_path)
        predictions = self.iron_model.predict(image_array, verbose=0)
        
        iron_prob = float(predictions[0][0])
        normal_prob = 1 - iron_prob
        
        return {
            'iron_probability': iron_prob,
            'normal_probability': normal_prob,
            'prediction': 'iron_deficiency' if iron_prob > 0.5 else 'normal'
        }

    def smart_routing_analysis(self, image_path):
        """Run both models and make smart decision"""
        
        # Get predictions from both models
        sickle_results = self.analyze_with_sickle_model(image_path)
        iron_results = self.analyze_with_iron_model(image_path)
        
        # Decision logic
        sickle_confidence = sickle_results['sickle_probability']
        iron_confidence = iron_results['iron_probability']
        
        # Thresholds for confident predictions
        high_confidence_threshold = 0.7
        
        results = {
            'sickle_analysis': sickle_results,
            'iron_analysis': iron_results,
        }
        
        # Smart routing logic
        if sickle_confidence > high_confidence_threshold and sickle_confidence > iron_confidence:
            final_diagnosis = 'sickle_cell'
            confidence = sickle_confidence
            vqa_type = 'sickle_cell'
            
        elif iron_confidence > high_confidence_threshold and iron_confidence > sickle_confidence:
            final_diagnosis = 'iron_deficiency'
            confidence = iron_confidence
            vqa_type = 'iron_deficiency'
            
        elif sickle_confidence > iron_confidence:
            if sickle_results['prediction'] == 'normal':
                final_diagnosis = 'likely_normal'
                confidence = sickle_results['normal_probability']
                vqa_type = 'none'
            else:
                final_diagnosis = 'possible_sickle_cell'
                confidence = sickle_confidence
                vqa_type = 'sickle_cell'
                
        else:
            if iron_results['prediction'] == 'normal':
                final_diagnosis = 'likely_normal'
                confidence = iron_results['normal_probability']
                vqa_type = 'none'
            else:
                final_diagnosis = 'possible_iron_deficiency'
                confidence = iron_confidence
                vqa_type = 'iron_deficiency'
        
        results['final_decision'] = {
            'diagnosis': final_diagnosis,
            'confidence': confidence,
            'vqa_type': vqa_type,
            'recommendation': self.get_recommendation(final_diagnosis, confidence)
        }
        
        return results

    def get_recommendation(self, diagnosis, confidence):
        """Get clinical recommendation based on results"""
        if 'normal' in diagnosis:
            return "No significant abnormalities detected"
        elif confidence > 0.8:
            return f"High confidence {diagnosis} detected - recommend clinical review"
        elif confidence > 0.6:
            return f"Moderate confidence {diagnosis} detected - suggest further testing"
        else:
            return f"Low confidence {diagnosis} detected - recommend additional analysis"

    def run_vqa_analysis(self, image_path, vqa_type, question):
        """Route VQA to the appropriate module based on condition detected"""
        try:
            # Debug info
            print(f"Debug - VQA Type: {vqa_type}")
            print(f"Debug - Sickle module loaded: {self.sickle_vqa_module is not None}")
            print(f"Debug - Iron module loaded: {self.iron_vqa_module is not None}")
            
            if vqa_type == "sickle_cell":
                if self.sickle_vqa_module is None:
                    return "Sickle cell VQA module not loaded. Check try1.py path.", None
                
                # Check what functions are available in the module
                available_functions = [func for func in dir(self.sickle_vqa_module) if not func.startswith('_')]
                print(f"Available functions in sickle module: {available_functions}")
                
                if hasattr(self.sickle_vqa_module, 'vqa_answer'):
                    # Get prediction from sickle model for VQA context
                    sickle_results = self.analyze_with_sickle_model(image_path)
                    label = "Sickle Cell" if sickle_results['sickle_probability'] > 0.5 else "Normal Cell"
                    confidence = sickle_results['sickle_probability'] if label == "Sickle Cell" else sickle_results['normal_probability']
                    
                    # Call the VQA function from try1.py
                    answer, processed_image = self.sickle_vqa_module.vqa_answer(question, image_path, label, confidence)
                    return answer, processed_image
                else:
                    return f"vqa_answer function not found in try1.py. Available: {available_functions}", None
            
            elif vqa_type == "iron_deficiency":
                if self.iron_vqa_module is None:
                    return "Iron deficiency VQA module not loaded. Check try2.py path.", None
                
                # Check what functions are available in the module
                available_functions = [func for func in dir(self.iron_vqa_module) if not func.startswith('_')]
                print(f"Available functions in iron module: {available_functions}")
                
                if hasattr(self.iron_vqa_module, 'vqa_answer'):
                    # Call the VQA function from try2.py
                    answer, processed_image = self.iron_vqa_module.vqa_answer(question, image_path)
                    return answer, processed_image
                else:
                    return f"vqa_answer function not found in try2.py. Available: {available_functions}", None
            
            elif vqa_type == "none":
                return "No VQA needed - image appears normal", None
            
            else:
                return f"Unknown VQA type: {vqa_type}. Expected: sickle_cell, iron_deficiency, or none", None
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return f"VQA Error: {str(e)}\nDetails: {error_details}", None

# ===================================================
# STREAMLIT APP WITH CONNECTED VQA FILES
# ===================================================

def main_app():
    """Streamlit app using connected VQA files"""
    st.title("🩸 Blood Smear Analysis System")
    st.markdown("*Upload a blood smear image for automatic analysis*")

    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        with st.spinner("Loading analysis models..."):
            st.session_state.analyzer = DualModelAnalyzer()

    uploaded_file = st.file_uploader("Upload a blood smear image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Blood Smear", use_container_width=True)
        
        # Save temporarily
        temp_path = f"temp_{uploaded_file.name}"
        image.save(temp_path)
        
        with st.spinner("Running analysis with both models..."):
            results = st.session_state.analyzer.smart_routing_analysis(temp_path)
        
        # Display results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.header("🔬 Sickle Cell Analysis")
            sickle = results['sickle_analysis']
            st.metric("Sickle Probability", f"{sickle['sickle_probability']:.2%}")
            st.metric("Normal Probability", f"{sickle['normal_probability']:.2%}")
            st.write(f"*Prediction:* {sickle['prediction']}")
        
        with col2:
            st.header("🩸 Iron Deficiency Analysis")
            iron = results['iron_analysis']
            st.metric("Iron Def. Probability", f"{iron['iron_probability']:.2%}")
            st.metric("Normal Probability", f"{iron['normal_probability']:.2%}")
            st.write(f"*Prediction:* {iron['prediction']}")
        
        with col3:
            st.header("🎯 Final Decision")
            final = results['final_decision']
            st.metric("Diagnosis", final['diagnosis'])
            st.metric("Confidence", f"{final['confidence']:.2%}")
            st.write(f"*VQA Available:* {final['vqa_type']}")
        
        # Recommendations
        st.header("📋 Clinical Recommendation")
        final = results['final_decision']
        
        if 'normal' in final['diagnosis']:
            st.success(f"✅ {final['recommendation']}")
        elif final['confidence'] > 0.8:
            st.error(f"🚨 {final['recommendation']}")
        elif final['confidence'] > 0.6:
            st.warning(f"⚠ {final['recommendation']}")
        else:
            st.info(f"ℹ {final['recommendation']}")
        
        # ===================================================
        # VQA SECTION - BASED ON FINAL DECISION
        # ===================================================
        vqa_type = results['final_decision']['vqa_type']
        
        if vqa_type == "sickle_cell":
            st.header("💬 Sickle Cell VQA - Ask Questions")
            st.info("Connected to try1.py - Ask questions about sickle cell analysis")
            
            
            if st.button("What causes sickle cell?", key="sickle_causes"):
                    question = "what causes sickle cell"
                    with st.spinner("Processing with sickle cell VQA (try1.py)..."):
                        answer, processed_img = st.session_state.analyzer.run_vqa_analysis(
                            temp_path, "sickle_cell", question
                        )
                        st.write("**Answer:**", answer)
                        if processed_img is not None:
                            st.image(processed_img, use_container_width=True)
            
            # Custom question input
            custom_question = st.text_input(
                "Ask your own question about sickle cell analysis:", 
                placeholder="e.g., What are the symptoms of sickle cell disease?",
                key="sickle_custom_input"
            )
            
            if custom_question and st.button("Get Answer", key="sickle_custom"):
                with st.spinner("Processing custom question with try1.py..."):
                    answer, processed_img = st.session_state.analyzer.run_vqa_analysis(
                        temp_path, "sickle_cell", custom_question
                    )
                    st.write("**Answer:**", answer)
                    if processed_img is not None:
                        st.image(processed_img, caption="Processed Image", use_container_width=True)

        elif vqa_type == "iron_deficiency":
            st.header("💬 Iron Deficiency VQA - Ask Questions")
            st.info("Connected to try2.py - Ask questions about iron deficiency analysis")
            
            # Pre-defined questions for easy access
            
            if st.button("What is IDA?", key="iron_definition"):
                    question = "what is IDA"
                    with st.spinner("Processing with iron deficiency VQA (try2.py)..."):
                        answer, processed_img = st.session_state.analyzer.run_vqa_analysis(
                            temp_path, "iron_deficiency", question
                        )
                        st.write("**Answer:**", answer)
                        if processed_img is not None:
                            st.image(processed_img,use_container_width=True)
            
            # Custom question input
            custom_question = st.text_input(
                "Ask your own question about iron deficiency analysis:",
                placeholder="e.g., What are the symptoms of iron deficiency?",
                key="iron_custom_input"
            )
            
            if custom_question and st.button("Get Answer", key="iron_custom"):
                with st.spinner("Processing custom question with try2.py..."):
                    answer, processed_img = st.session_state.analyzer.run_vqa_analysis(
                        temp_path, "iron_deficiency", custom_question
                    )
                    st.write("**Answer:**", answer)
                    if processed_img is not None:
                        st.image(processed_img, caption="Processed Image", use_container_width=True)
        
        elif vqa_type == "none":
            st.header("💬 VQA Status")
            st.success("✅ No abnormalities detected - VQA not needed")
            st.markdown("""
            **Why no VQA?**
            - Image appears normal with high confidence
            - No significant abnormalities requiring detailed analysis
            - VQA will activate automatically if abnormalities are detected
            
            **Available VQA Systems:**
            - 🔬 Sickle Cell VQA (try1.py) - Activates for sickle cell detection
            - 🩸 Iron Deficiency VQA (try2.py) - Activates for IDA detection
            """)
        
        # Show detailed analysis in expander
        with st.expander("🔍 Detailed Technical Analysis"):
            st.subheader("Model Outputs")
            st.json(results)
            
            st.subheader("VQA Module Status")
            sickle_status = "✅ Connected" if st.session_state.analyzer.sickle_vqa_module else "❌ Not loaded"
            iron_status = "✅ Connected" if st.session_state.analyzer.iron_vqa_module else "❌ Not loaded"
            
            st.write(f"**Sickle Cell VQA (try1.py):** {sickle_status}")
            st.write(f"**Iron Deficiency VQA (try2.py):** {iron_status}")
        
        # Cleanup
        try:
            os.remove(temp_path)
        except:
            pass

    else:
        st.info("Please upload a blood smear image to get started!")
        
        # Show simple information about the system
        st.markdown("### How it works:")
        st.markdown("""
        1. Upload a blood smear image
        2. Automatic analysis for blood disorders  
        3. Get results and interactive Q&A if needed
        """)

        # Note about what happens after upload
        st.markdown("### What happens next:")
        st.markdown("""
        - Image will be analyzed automatically
        - Results will show any detected conditions
        - Question & Answer interface will appear based on findings
        - No Q&A interface for normal/healthy samples
        """)

# ===================================================
# MAIN EXECUTION
# ===================================================

if __name__ == "__main__":
    main_app()