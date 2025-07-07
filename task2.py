import speech_recognition as sr


def take_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"You said: {query}")
    except Exception:
        print("Sorry, I didn't catch that.")
        return "None"
    return query.lower()

def main():
    print("Listening...")
    take_command()

if __name__ == "__main__":
    main()