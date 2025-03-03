# 🌱 AI Plant Recognition Program

## 📁 Folder Structure
```
main/
│── datasets/
│   ├── plants/
│   │   ├── Rose/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   ├── ...
│   │   ├── Tulip/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   ├── ...
│   │   ├── Sunflower/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   ├── ...
│   │   ├── ... (Other plant species)
│── train.py
│── plantsModel.pkl  (Generated after training and renamed)
│── ...
```

---

## 🚀 Installation & Setup

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/H1NORI/plant-recognition.git
cd plant-recognition
```

### 2️⃣ **Prepare Dataset**
Place images of different plant species into separate folders inside `main/datasets/plants/`.

Ensure each folder name corresponds to the species name.

### 3️⃣ **Train the Model**
```bash
python main/train.py
```

- The training script will process the dataset and generate a model file `model.pkl`.
- Rename the model file after training:
```bash
mv main/model.pkl main/plantsModel.pkl
```

---

## 🎯 Run the Main Program
After training, use the trained model to classify plant images:
```bash
python manage.py runserver
```

This will load `plantsModel.pkl` and allow plant image classification by running webserver.

Now, open your browser and go to:
http://localhost:8800

---


