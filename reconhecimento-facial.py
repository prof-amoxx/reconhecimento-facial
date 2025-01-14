#Sistema de Reconhecimento Facial com TensorFlow

!pip install tensorflow
!pip install opencv-python
!pip install mtcnn
!pip install keras
!pip install numpy
  
#Importação das Bibliotecas e Configuração
import tensorflow as tf
import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuração de GPU se disponível
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
  
#Detector Facial usando MTCNN
class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()
        
    def detect_faces(self, image):
        # Detecta faces na imagem
        faces = self.detector.detect_faces(image)
        return faces

    def extract_faces(self, image, required_size=(160, 160)):
        faces = self.detect_faces(image)
        face_images = []
        face_boxes = []
        
        for face in faces:
            x1, y1, width, height = face['box']
            x2, y2 = x1 + width, y1 + height
            
            # Extrai a face
            face_boundary = image[y1:y2, x1:x2]
            
            # Redimensiona a face
            face_image = cv2.resize(face_boundary, required_size)
            
            face_images.append(face_image)
            face_boxes.append((x1, y1, x2, y2))
            
        return face_images, face_boxes
  
#Modelo de Reconhecimento Facial
def create_recognition_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model
  
#Sistema Completo de Detecção e Reconhecimento
class FaceRecognitionSystem:
    def __init__(self, model_path=None, class_names=None):
        self.detector = FaceDetector()
        if model_path:
            self.recognition_model = load_model(model_path)
            self.class_names = class_names
        else:
            self.recognition_model = None
            self.class_names = None

    def process_image(self, image):
        # Detecta e extrai faces
        face_images, face_boxes = self.detector.extract_faces(image)
        
        results = []
        for face_img, box in zip(face_images, face_boxes):
            # Pré-processamento da face
            face_array = np.asarray(face_img)
            face_array = face_array.astype('float32')
            face_array = face_array / 255.0
            face_array = np.expand_dims(face_array, axis=0)
            
            # Reconhecimento
            if self.recognition_model:
                prediction = self.recognition_model.predict(face_array)
                class_idx = np.argmax(prediction[0])
                confidence = prediction[0][class_idx]
                name = self.class_names[class_idx]
            else:
                name = "Unknown"
                confidence = 0.0
                
            results.append({
                'box': box,
                'name': name,
                'confidence': confidence
            })
            
        return results

    def draw_results(self, image, results):
        for result in results:
            x1, y1, x2, y2 = result['box']
            name = result['name']
            confidence = result['confidence']
            
            # Desenha a caixa
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Adiciona o texto
            text = f"{name} ({confidence:.2f})"
            cv2.putText(image, text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        return image
  

# Inicialização do sistema
face_system = FaceRecognitionSystem('model.h5', ['Person1', 'Person2', 'Person3'])

# Captura de vídeo em tempo real
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Processo de detecção e reconhecimento
    results = face_system.process_image(frame)
    
    # Desenha os resultados
    frame = face_system.draw_results(frame, results)
    
    # Mostra o resultado
    cv2.imshow('Face Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
  
