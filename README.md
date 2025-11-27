# Smart-spectacles-
Smart spectacles for visually impaired individuals 
import cv2
import time
import logging
from PIL import Image
import pyttsx3
import threading
import atexit
import os
import numpy as np
import torch
import easyocr
import re
import nltk
from nltk.tokenize import sent_tokenize
import speech_recognition as sr
import pickle
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# Download NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except ModuleNotFoundError:
    print("Error: Required module 'transformers' is not installed. Please install it using 'pip install transformers' and try again.")
    exit()


class SmartGlassesSystem:
    def __init__(self):
        self.tts_engine = self.init_tts()
        self.tts_lock = threading.Lock()
        self.last_tts_time = time.time()
        self.object_model = self.load_object_detection_model()
        self.captioning_model, self.captioning_processor = self.load_captioning_model()
        self.camera_url = 0
        self.last_capture_time = time.time()
        self.capture_interval = 11
        self.mode = 'description'
        self.feedback_mode = 'contextual'
        self.navigation_active = False
        self.reading_confidence_threshold = 0.7
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        self.running = True
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.voice_command_thread = None

        # Initialize face recognition with proper configuration
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Load face database - matches the format from your training code
        self.face_model_path = "face_model.pkl"
        self.known_faces = self.load_face_database()

        atexit.register(self.cleanup)

    def load_face_database(self):
        """Load face database matching the format from training code"""
        if os.path.exists(self.face_model_path):
            try:
                with open(self.face_model_path, 'rb') as f:
                    known_faces = pickle.load(f)
                print(f"Loaded face database with {len(known_faces)} known people")
                # Convert to list format if it's in dictionary format
                if isinstance(known_faces, dict):
                    return known_faces
                else:
                    return {}
            except Exception as e:
                print(f"Error loading face database: {e}")
                return {}
        return {}

    def init_tts(self):
        """Initialize text-to-speech engine"""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 0.9)
            voices = engine.getProperty('voices')
            for voice in voices:
                if "premium" in voice.name.lower() or "natural" in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            return engine
        except Exception as e:
            logging.error(f"TTS Initialization Error: {e}")
            raise

    def speak(self, text, cooldown=1.5):
        """Convert text to speech with cooldown"""
        current_time = time.time()
        if current_time - self.last_tts_time < cooldown:
            return
        self.last_tts_time = current_time

        def run_tts():
            with self.tts_lock:
                try:
                    processed_text = self.process_text_for_tts(text)
                    self.tts_engine.say(processed_text)
                    self.tts_engine.runAndWait()
                except Exception as e:
                    logging.error(f"TTS Error: {e}")

        threading.Thread(target=run_tts).start()

    def process_text_for_tts(self, text):
        """Process text for more natural speech output"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = sent_tokenize(text)
        processed_sentences = []
        for sentence in sentences:
            if sentence and len(sentence) > 0:
                processed_sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence[0].upper()
                if not processed_sentence[-1] in ['.', '!', '?']:
                    processed_sentence += '.'
                processed_sentences.append(processed_sentence)
        processed_text = ' '.join(processed_sentences)
        processed_text = processed_text.replace(':', ', ')
        processed_text = re.sub(r'\bDr\.', 'Doctor', processed_text)
        processed_text = re.sub(r'\bMr\.', 'Mister', processed_text)
        processed_text = re.sub(r'\bMrs\.', 'Misses', processed_text)
        processed_text = re.sub(r'\bMs\.', 'Miss', processed_text)
        processed_text = re.sub(r'\bSt\.', 'Street', processed_text)
        processed_text = re.sub(r'\bAve\.', 'Avenue', processed_text)
        processed_text = re.sub(r'\bRd\.', 'Road', processed_text)
        return processed_text

    def load_object_detection_model(self):
        """Load YOLO object detection model"""
        try:
            model = YOLO("yolov8n.pt")
            if torch.cuda.is_available():
                model.to("cuda")
                logging.info("YOLO model moved to GPU.")
            else:
                logging.warning("GPU not available. YOLO model will run on CPU.")
            return model
        except Exception as e:
            logging.error(f"Object Detection Model Loading Error: {e}")
            raise

    def load_captioning_model(self):
        """Load BLIP image captioning model"""
        try:
            model_folder = "D:\ALEX\COLLEGE\MINIPROJECT\model"
            processor = BlipProcessor.from_pretrained(model_folder)
            model = BlipForConditionalGeneration.from_pretrained(model_folder)
            if torch.cuda.is_available():
                model.to("cuda")
                logging.info("BLIP model moved to GPU.")
            else:
                logging.warning("GPU not available. BLIP model will run on CPU.")
            return model, processor
        except Exception as e:
            logging.error(f"Image Captioning Model Loading Error: {e}")
            raise

    def detect_objects(self, frame):
        """Detect objects in frame using YOLO"""
        try:
            results = self.object_model(frame)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            return boxes, classes, scores
        except Exception as e:
            logging.error(f"Object Detection Error: {e}")
            return [], [], []

    def generate_caption(self, frame):
        """Generate image caption using BLIP model"""
        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self.captioning_processor(images=image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            out = self.captioning_model.generate(**inputs)
            caption = self.captioning_processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            logging.error(f"Captioning Error: {e}")
            return ""

    def describe_surroundings(self, frame):
        """Describe the current surroundings"""
        boxes, classes, scores = self.detect_objects(frame)
        object_counts = {}
        for box, class_id, score in zip(boxes, classes, scores):
            object_name = self.object_model.names[int(class_id)]
            if object_name not in object_counts:
                object_counts[object_name] = 0
            object_counts[object_name] += 1

        description_text = ""
        if self.feedback_mode == 'minimal':
            if object_counts:
                description_text += "You see: " + ", ".join([f"{count} {name}{'s' if count > 1 else ''}" for name, count in object_counts.items()])
            else:
                description_text += "No objects detected in view."
        elif self.feedback_mode == 'contextual':
            caption = self.generate_caption(frame)
            description_text += caption if caption else "Unable to describe the current scene."

        self.speak(description_text)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame

    def recognize_faces(self, frame):
        """Recognize faces in the frame"""
        try:
            faces = self.face_app.get(frame)
            recognized_names = []
            
            for face in faces:
                if face.det_score < 0.5:  # Skip low confidence detections
                    continue
                    
                # Get face embedding
                embedding = face.embedding
                
                # Find best match
                best_match = None
                best_similarity = 0.7  # Minimum similarity threshold
                
                for name, embeddings in self.known_faces.items():
                    for known_embedding in embeddings:
                        similarity = np.dot(known_embedding, embedding) / (
                            np.linalg.norm(known_embedding) * np.linalg.norm(embedding))
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = name
                
                if best_match:
                    recognized_names.append(best_match)
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, best_match, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Generate description
            if recognized_names:
                names = ", ".join(recognized_names)
                description = f"I see {names}."
                
                # Add contextual information
                boxes, classes, _ = self.detect_objects(frame)
                object_names = [self.object_model.names[int(cls)] for cls in classes]
                
                if 'chair' in object_names or 'sofa' in object_names:
                    description += " They are sitting down."
                elif 'table' in object_names:
                    description += " They are near a table."
                    
                self.speak(description)
            else:
                self.speak("I don't recognize anyone here.")

            return frame
        except Exception as e:
            logging.error(f"Face recognition error: {e}")
            self.speak("Error recognizing faces.")
            return frame

    def read_text(self, frame):
        """Read text from the frame using EasyOCR"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = self.reader.readtext(gray)
            valid_results = [r for r in results if r[2] >= self.reading_confidence_threshold]
            if valid_results:
                sorted_results = sorted(valid_results, key=lambda r: (r[0][0][1] + r[0][2][1]) / 2)
                line_threshold = 20
                lines = []
                current_line = [sorted_results[0]]
                for i in range(1, len(sorted_results)):
                    current_y = (sorted_results[i][0][0][1] + sorted_results[i][0][2][1]) / 2
                    prev_y = (current_line[-1][0][0][1] + current_line[-1][0][2][1]) / 2
                    if abs(current_y - prev_y) < line_threshold:
                        current_line.append(sorted_results[i])
                    else:
                        current_line.sort(key=lambda r: r[0][0][0])
                        lines.append(current_line)
                        current_line = [sorted_results[i]]
                if current_line:
                    current_line.sort(key=lambda r: r[0][0][0])
                    lines.append(current_line)
                text_by_lines = []
                for line in lines:
                    line_text = " ".join([r[1] for r in line])
                    text_by_lines.append(line_text)
                complete_text = ". ".join(text_by_lines)
                if complete_text:
                    for (bbox, text, prob) in valid_results:
                        (top_left, top_right, bottom_right, bottom_left) = bbox
                        top_left = (int(top_left[0]), int(top_left[1]))
                        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                    self.speak(f"I found the following text: {complete_text}")
                else:
                    self.speak("No readable text detected.")
            else:
                self.speak("No readable text found in view.")
            cv2.putText(frame, "Reading Mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame
        except Exception as e:
            logging.error(f"Text Reading Error: {e}")
            self.speak("I had trouble reading text in this image.")
            return frame

    def navigate(self, frame):
        """Navigation assistance mode"""
        boxes, classes, scores = self.detect_objects(frame)
        
        obstacle_classes = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'chair', 'table']
        obstacles = []
        for box, class_id, score in zip(boxes, classes, scores):
            class_name = self.object_model.names[int(class_id)]
            if class_name in obstacle_classes and score > 0.5:
                obstacles.append((box, class_name))
        
        if obstacles:
            obstacle_text = ", ".join([name for _, name in obstacles])
            self.speak(f"Caution! There are {obstacle_text} ahead.")
        else:
            self.speak("The path is clear.")
        
        for box, _ in obstacles:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        return frame

    def process_frame(self, frame):
        """Process frame based on current mode"""
        if time.time() - self.last_capture_time < self.capture_interval:
            return frame
        self.last_capture_time = time.time()

        if self.mode == 'description':
            return self.describe_surroundings(frame)
        elif self.mode == 'navigation':
            return self.navigate(frame)
        elif self.mode == 'reading':
            return self.read_text(frame)
        elif self.mode == 'face_recognition':
            return self.recognize_faces(frame)
        return frame

    def display_menu(self):
        """Display system menu options"""
        menu_text = "Smart Glasses activated. Available modes are: Description, Navigation, Reading, and Face Recognition. Press D for description, N for navigation, R for reading mode, F for face recognition, or H for help."
        self.speak(menu_text)
        print("\n===== SMART GLASSES SYSTEM =====")
        print("Options:")
        print("[d] - Environment Description Mode")
        print("[n] - Navigation Mode")
        print("[r] - Reading Mode")
        print("[f] - Face Recognition Mode")
        print("[h] - Help")
        print("[q] - Quit")
        print("\nFeedback Modes:")
        print("[1] - Minimal")
        print("[2] - Contextual")
        print("================================\n")

    def start_voice_command_listener(self):
        """Start listening for voice commands in background"""
        def listen_for_commands():
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                while self.running:
                    try:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                        command = self.recognizer.recognize_google(audio).lower()
                        logging.info(f"Detected voice command: {command}")
                        if "description" in command:
                            self.mode = 'description'
                            self.speak("Switching to description mode.")
                        elif "navigation" in command:
                            self.mode = 'navigation'
                            self.speak("Switching to navigation mode.")
                        elif "reading" in command:
                            self.mode = 'reading'
                            self.speak("Switching to reading mode.")
                        elif "face recognition" in command:
                            self.mode = 'face_recognition'
                            self.speak("Switching to face recognition mode.")
                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        logging.warning("Could not understand the audio.")
                    except sr.RequestError as e:
                        logging.error(f"Speech recognition error: {e}")

        self.voice_command_thread = threading.Thread(target=listen_for_commands, daemon=True)
        self.voice_command_thread.start()

    def cleanup(self):
        """Clean up system resources"""
        logging.info("Cleaning up resources...")
        if hasattr(self, 'tts_engine'):
            try:
                self.tts_engine.stop()
            except Exception as e:
                logging.error(f"Error stopping TTS engine: {e}")
        
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logging.error(f"Error closing OpenCV windows: {e}")
            
        logging.info("Cleanup complete.")

    def run(self):
        """Main system execution loop"""
        try:
            self.start_voice_command_listener()
            cap = cv2.VideoCapture(self.camera_url)
            if not cap.isOpened():
                logging.warning(f"Camera {self.camera_url} connection failed, trying default camera")
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    self.speak("Camera connection failed. Please check your camera and try again.")
                    return
                else:
                    self.speak("Connected to default camera.")
            else:
                self.speak("Camera connected successfully.")

            self.display_menu()

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logging.error("Frame capture error")
                    self.speak("I'm having trouble getting camera feed.")
                    time.sleep(1)
                    continue

                processed_frame = self.process_frame(frame)
                if processed_frame is not None:
                    cv2.imshow("Smart Glasses View", processed_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    self.speak("Shutting down smart glasses system.")
                    break
                elif key == ord('f'):
                    self.mode = 'face_recognition'
                    self.speak("Switching to face recognition mode.")

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            self.speak("An error occurred. The system needs to restart.")
        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()
            self.cleanup()
            logging.info("Program exited cleanly.")
            os._exit(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting Smart Glasses System...")
    SmartGlassesSystem().run()