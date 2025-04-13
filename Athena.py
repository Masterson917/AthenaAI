import pyttsx3
import speech_recognition as sr
import webbrowser
import random
from PIL import Image
from transformers import GPTJForCausalLM, GPT2Tokenizer

# Initialize the TTS engine
engine = pyttsx3.init()

# Set voice to female
voices = engine.getProperty('voices')
for voice in voices:
    if "female" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break

def speak(text):
    print(f"Athena 🗣️: {text}")
    engine.say(text)
    engine.runAndWait()

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"🗣️ You said: {text}")
        return text
    except sr.UnknownValueError:
        print("❌ Sorry, I couldn't understand that.")
        return ""
    except sr.RequestError:
        print("❌ Speech recognition service error.")
        return ""

def get_text_input():
    return input("⌨️ Type your message: ")

def get_user_input():
    method = input("Choose input method ([v]oice / [t]ext): ").strip().lower()
    if method == "v":
        return get_voice_input()
    else:
        return get_text_input()

# Load GPT-J model and tokenizer
model_name = "EleutherAI/gpt-j-6B"
model = GPTJForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def get_openai_response(prompt):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate a response from the model
    outputs = model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)
    
    # Decode and return the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

def handle_command(command):
    command = command.lower()

    if "open google" in command:
        speak("Opening Google.")
        webbrowser.open("https://www.google.com")
    elif "random number" in command:
        number = random.randint(1, 100)
        speak(f"Your random number is {number}")
    elif "show image" in command:
        try:
            img = Image.open("sample.jpg")  # Replace with your image path
            img.show()
            speak("Here is the image.")
        except Exception as e:
            speak(f"Couldn't open the image: {str(e)}")
    elif "exit" in command:
        speak("Goodbye!")
        return False
    else:
        response = get_openai_response(command)
        speak(response)
    
    return True

# Main program loop
if __name__ == "__main__":
    speak("Hi, I'm Athena. How can I help you today?")

    while True:
        user_input = get_user_input()
        if not user_input:
            continue
        if not handle_command(user_input):
            break
