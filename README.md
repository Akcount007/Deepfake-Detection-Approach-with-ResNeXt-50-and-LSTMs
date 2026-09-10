
# 🕵️‍♂️ Discerning Deception: Face-Centric Deepfake Detection-Approach-with-ResNeXt-50-and-LSTMs

![Status](https://img.shields.io/badge/Status-Experimental_Research-blue?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=for-the-badge&logo=pytorch)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8?style=for-the-badge&logo=opencv)

> **"Seeing is no longer believing, but our experimental model explores ways to tell the difference."**

Welcome to an experimental, dual-stream deep learning architecture designed to study and unmask digital forgery[cite: 1, 2]. As Generative Adversarial Networks (GANs) and autoencoders become increasingly capable of generating believable deepfakes, malicious actors frequently misuse these tools to manipulate media[cite: 1, 2]. This project presents a pipeline that combines the spatial feature extraction of **ResNeXt-50** with the temporal sequence modeling of **Long Short-Term Memory (LSTM)** networks to study manipulated content[cite: 1, 2]. 

*Disclaimer: This repository serves as an academic and theoretical exploration of deepfake detection. The performance metrics detailed below reflect observed experimental results within a specific testing environment and do not constitute absolute claims of real-world infallibility.*

---

## 🧠 The Architecture: Space & Time

Deepfakes often leave behind microscopic clues in a single image, as well as unnatural transitions over time. This architecture tackles the problem in two experimental stages:

1. 🖼️ **The Spatial Inspector (ResNeXt-50):**
   The video is initially split into frames, and a robust face detection algorithm isolates and crops the facial regions[cite: 1, 2]. A pre-trained ResNeXt-50 (32x4d) architecture processes these face-only frames, extracting 2048-dimensional spatial feature vectors that capture subtle visual anomalies and blending errors[cite: 1, 2].  

2. ⏱️ **The Temporal Detective (LSTM):**
   Because deepfakes often exhibit temporal inconsistencies, the extracted spatial features are passed into an LSTM network[cite: 1, 2]. The LSTM analyzes a consecutive sequence of 100 frames to track the evolution of facial expressions and movements over time[cite: 1, 2].  

---

## 🏆 Performance & Experimental Metrics

Evaluated against a robust mixture of authentic and fake videos during the study, the model achieved highly competitive accuracy metrics within its test environment[cite: 1, 2].  

*   **Overall Accuracy:** The model achieved an observed 94% overall accuracy in distinguishing real from fake videos[cite: 1, 2].  
*   **Real Video Confidence:** When predicting authentic, unmodified footage during the testing phase, the model operated with 95% confidence accuracy[cite: 1, 2].  
*   **Fake Video Confidence:** When exposed to a deepfake manipulation of the same individual within the test set, the model correctly flagged it with 96% confidence[cite: 1, 2].  

---

## 💻 Prerequisites & Required Libraries

To replicate this experimental setup, specific hardware and software configurations are required due to the heavy computational demands of processing video batches[cite: 1, 2].

### Hardware Minimum Requirements
*   **Processor:** Intel Xeon E5 2637 (3.5 GHz)[cite: 1]
*   **Memory (RAM):** 8 GB[cite: 1]
*   **Storage:** 150 GB Hard Disk[cite: 1]
*   **GPU:** NVIDIA GeForce (4 GB RAM)[cite: 1]

### Software & Library Dependencies
*   **Operating System:** Windows 7+[cite: 1]
*   **Programming Language:** [Python 3.0](https://www.python.org/)[cite: 1]
*   **Deep Learning Framework:** [PyTorch 1.4](https://pytorch.org/)[cite: 1] (Required for robust CUDA/GPU support)
*   **Computer Vision Library:** [OpenCV](https://opencv.org/)[cite: 1] (Used for video splitting and face cropping)
*   **Face Detection Library:** [Face-recognition](https://github.com/ageitgey/face_recognition)[cite: 1]

---

## 🗄️ The Data Arsenal (Datasets)

To ensure the model was exposed to diverse deepfake generation techniques and authentic situations, it was trained on an amalgamation of the industry's most challenging benchmarks[cite: 1, 2]:  

🌟 **FaceForensics++ (FF++)**[cite: 1, 2]
*   **Overview:** Contains a massive repository of 4000 total videos (3000 fake, 1000 real)[cite: 1, 2].  
*   **Characteristics:** Poses a great challenge due to the sheer variety of deepfake types and temporal flickering, though videos are limited in length[cite: 1, 2]. 
*   **Resource Link:** [FaceForensics GitHub Repository](https://github.com/ondyari/FaceForensics)

🛡️ **Deepfake Detection Challenge (DFDC)**[cite: 1, 2]
*   **Overview:** A large-scale dataset featuring 5250 total videos (4119 fake, 1131 real)[cite: 1, 2].  
*   **Characteristics:** Features highly improved visual quality and temporal consistency, representing diverse creation techniques[cite: 1, 2]. 
*   **Resource Link:** [DFDC Meta AI Portal](https://ai.meta.com/datasets/dfdc/)

🎭 **Celeb-DF**[cite: 1, 2]
*   **Overview:** Comprises 1203 high-quality videos (795 fake, 408 real) focused on celebrity and YouTuber facial alterations[cite: 1, 2].  
*   **Characteristics:** Offers straightforward access to improved, highly convincing visual manipulations[cite: 1, 2].  
*   **Resource Link:** [Celeb-DF GitHub Repository](https://github.com/yuezunli/celeb-deepfakeforensics)

*(Note: To further enrich the training process, the dataset utilizes arbitrary cropping, image rotation, mirroring, and color tone variations to prevent overfitting[cite: 1, 2].)*  

---

## 🚀 Getting Started

*(Note: The full project files are provided directly in this directory. Ensure all local scripts and files are downloaded to begin.)*

### 1. Prepare Your Dataset
Organize your chosen dataset into a training (80%) and testing (20%) split[cite: 1, 2].
During preprocessing, a threshold value ($p$) is calculated to ensure a uniform number of frames across videos[cite: 1, 2]:

$p=mean(\{k_j\}|j=1~to~a)$[cite: 1, 2]

*(To optimize for GPU limitations, the system crops and retains a maximum of 200 frames per video at 30 FPS[cite: 1, 2].)*  

### 2. Training the Model
The model utilizes hyperparameter tuning for exceptional performance[cite: 1, 2]. By default, it uses the Adam optimizer with a learning rate and weight decay set at $1\times10^{-5}$, operating at a batch size of 4[cite: 1, 2].  

The model evaluates errors using the binary cross-entropy loss function[cite: 2]:

$L=-(1/N)*\sum_{j=1}^{N}(y_j*\ln(p_j)+(1-y_j)*\ln(1-p_j))$[cite: 2]

Execute your local training script. *Note: Early stopping is implemented; training will halt automatically when validation loss stops improving to prevent the model from overfitting[cite: 1, 2].*  

### 3. Catch a Fake (Inference)
Upload a user video to the experimental processing pipeline[cite: 1, 2]. The system will automatically detect faces, crop the region, and pass the sequence to the trained model to output a **REAL** or **FAKE** prediction[cite: 1, 2].
