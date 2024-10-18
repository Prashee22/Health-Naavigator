import gradio as gr
import torch
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from model import RNN_model
import nltk_utils
import time

# Setup class names
class_names = {
    0: 'Acne', 1: 'Arthritis', 2: 'Bronchial Asthma', 3: 'Cervical spondylosis',
    4: 'Chicken pox', 5: 'Common Cold', 6: 'Dengue', 7: 'Dimorphic Hemorrhoids',
    8: 'Fungal infection', 9: 'Hypertension', 10: 'Impetigo', 11: 'Jaundice',
    12: 'Malaria', 13: 'Migraine', 14: 'Pneumonia', 15: 'Psoriasis',
    16: 'Typhoid', 17: 'Varicose Veins', 18: 'allergy', 19: 'diabetes',
    20: 'drug reaction', 21: 'gastroesophageal reflux disease',
    22: 'peptic ulcer disease', 23: 'urinary tract infection'
}

# Disease Advice
disease_advice = {
    'Acne': "Maintain a proper skincare routine, avoid excessive touching of the affected areas, and consider using over-the-counter topical treatments. If severe, consult a dermatologist.",
    'Arthritis': "Stay active with gentle exercises, manage weight, and consider pain-relief strategies like hot/cold therapy. Consult a rheumatologist for tailored guidance.",
    'Bronchial Asthma': "Follow prescribed inhaler and medication regimen, avoid triggers like smoke and allergens, and have an asthma action plan. Regular check-ups with a pulmonologist are important.",
    'Cervical spondylosis': "Maintain good posture, do neck exercises, and use ergonomic support. Physical therapy and pain management techniques might be helpful.",
    'Chicken pox': "Rest, maintain hygiene, and avoid scratching. Consult a doctor for appropriate antiviral treatment.",
    'Common Cold': "Get plenty of rest, stay hydrated, and consider over-the-counter remedies for symptom relief. Seek medical attention if symptoms worsen or last long.",
    'Dengue': "Stay hydrated, rest, and manage fever with acetaminophen. Seek medical care promptly, as dengue can escalate quickly.",
    'Dimorphic Hemorrhoids': "Follow a high-fiber diet, maintain good hygiene, and consider stool softeners. Consult a doctor if symptoms persist.",
    'Fungal infection': "Keep the affected area clean and dry, use antifungal creams, and avoid sharing personal items. Consult a dermatologist if it persists.",
    'Hypertension': "Follow a balanced diet, exercise regularly, reduce salt intake, and take prescribed medications. Regular check-ups with a healthcare provider are important.",
    'Impetigo': "Keep the affected area clean, use prescribed antibiotics, and avoid close contact. Consult a doctor for proper treatment.",
    'Jaundice': "Get plenty of rest, maintain hydration, and follow a doctor's advice for diet and medications. Regular monitoring is important.",
    'Malaria': "Take prescribed antimalarial medications, rest, and manage fever. Seek medical attention for severe cases.",
    'Migraine': "Identify triggers, manage stress, and consider pain-relief medications. Consult a neurologist for personalized management.",
    'Pneumonia': "Follow prescribed antibiotics, rest, stay hydrated, and monitor symptoms. Seek immediate medical attention for severe cases.",
    'Psoriasis': "Moisturize, use prescribed creams, and avoid triggers. Consult a dermatologist for effective management.",
    'Typhoid': "Take prescribed antibiotics, rest, and stay hydrated. Dietary precautions are important. Consult a doctor for proper treatment.",
    'Varicose Veins': "Elevate legs, exercise regularly, and wear compression stockings. Consult a vascular specialist for evaluation and treatment options.",
    'allergy': "Identify triggers, manage exposure, and consider antihistamines. Consult an allergist for comprehensive management.",
    'diabetes': "Follow a balanced diet, exercise, monitor blood sugar levels, and take prescribed medications. Regular visits to an endocrinologist are essential.",
    'drug reaction': "Discontinue the suspected medication, seek medical attention if symptoms are severe, and inform healthcare providers about the reaction.",
    'gastroesophageal reflux disease': "Follow dietary changes, avoid large meals, and consider medications. Consult a doctor for personalized management.",
    'peptic ulcer disease': "Avoid spicy and acidic foods, take prescribed medications, and manage stress. Consult a gastroenterologist for guidance.",
    'urinary tract infection': "Stay hydrated, take prescribed antibiotics, and maintain good hygiene. Consult a doctor for appropriate treatment."
}

responses = {
    'greeting': [
        "Thank you for using our medical chatbot. Please provide the symptoms you're experiencing, and I'll do my best to predict the possible disease.",
        "Hello! I'm here to help you with medical predictions based on your symptoms. Please describe your symptoms in as much detail as possible.",
        "Greetings! I am a specialized medical chatbot trained to predict potential diseases based on the symptoms you provide. Kindly list your symptoms explicitly.",
        "Welcome to the medical chatbot. To assist you accurately, please share your symptoms in explicit detail.",
        "Hi there! I'm a medical chatbot specialized in analyzing symptoms to suggest possible diseases. Please provide your symptoms explicitly.",
    ],
    'goodbye': [
        "Take care of yourself! If you have more questions, don't hesitate to reach out.",
        "Stay well! Remember, I'm here if you need further medical advice.",
        "Goodbye for now! Don't hesitate to return if you need more information in the future.",
        "Wishing you good health ahead! Feel free to come back if you have more concerns.",
        "Farewell! If you have more symptoms or questions, don't hesitate to consult again.",
    ]
}

def preprocess_data():
    df = pd.read_csv('Symptom2Disease.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    train_data, test_data = train_test_split(df, test_size=0.15, random_state=42)
    return train_data, test_data

def load_model():
    model = RNN_model()
    model.load_state_dict(torch.load('pretrained_symtom_to_disease_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def get_vectorizer(train_data):
    vectorizer = nltk_utils.vectorizer()
    vectorizer.fit(train_data.text)
    return vectorizer

train_data, _ = preprocess_data()
vectorizer = get_vectorizer(train_data)
model = load_model()

def predict_disease(message):
    try:
        transform_text = vectorizer.transform([message])
        transform_text = torch.tensor(transform_text.toarray()).to(torch.float32)
        with torch.no_grad():
            y_logits = model(transform_text)
            pred_prob = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
        return class_names[pred_prob.item()]
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def respond(message, chat_history):
    message = message.lower()
    if any(greeting in message for greeting in ['hello', 'hi', 'hey', 'greetings']):
        bot_message = random.choice(responses['greeting'])
    elif any(goodbye in message for goodbye in ['bye', 'goodbye', 'farewell', 'thanks']):
        bot_message = random.choice(responses['goodbye'])
    else:
        predicted_disease = None
        for disease in class_names.values():
            if disease.lower() in message:
                predicted_disease = disease
                break
        
        if predicted_disease:
            bot_message = f"Based on your symptoms, I believe you are having {predicted_disease}. {disease_advice[predicted_disease]}"
        else:
            predicted_disease = predict_disease(message)
            if predicted_disease:
                bot_message = f"Based on your symptoms, I believe you may have {predicted_disease}. {disease_advice[predicted_disease]}"
            else:
                bot_message = "I'm sorry, I couldn't predict a disease based on the given symptoms. Please provide more details or consult a healthcare professional."

    chat_history.append((message, bot_message))
    time.sleep(2)  # Simulate processing time
    return "", chat_history

def main():
    with gr.Blocks(css="#col_container { margin-left: auto; margin-right: auto;} #chatbot {height: 520px; overflow: auto;}") as demo:
        gr.HTML('<h1 align="center">Medical Chatbot 🤖</h1>')
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
    demo.launch()

if __name__ == "__main__":
    main()