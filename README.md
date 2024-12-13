# Jacket and Helmet Detection System  

## Project Overview  
This project aims to develop an AI-based system that detects and classifies persons wearing helmets, safety jackets, both, or neither. It is intended to enhance safety compliance monitoring in industrial or construction environments. The system was implemented as part of a broader exploration of advanced machine learning techniques, including the YOLOv11 model for object detection.  

### Objectives  
1. Detect persons in images or videos.  
2. Classify individuals into four categories:  
   - Wearing only helmets.  
   - Wearing only safety jackets.  
   - Wearing both helmets and safety jackets.  
   - Wearing neither helmet nor jacket.  
3. Perform robust annotation to ensure accurate model training and evaluation.  
4. Train a custom YOLOv11 model to achieve high accuracy in detection and classification.  
5. Perform inference on custom datasets to demonstrate real-world applicability.  

---

## Requirements  
The following tools and technologies were used:  
1. **Ubuntu OS**: Preferred for its robustness in handling AI workflows.  
2. **Docker**: Used to containerize the development environment.  
3. **Kubeflow**: A comprehensive platform for developing and deploying AI models.  
4. **YOLOv11**: The object detection model used for this project, providing real-time detection capabilities.  
5. **Roboflow**: Used for dataset annotation and preprocessing.  
6. **PyTorch**: The deep learning framework powering YOLOv11.  
7. **Python**: Programming language for scripting and training.  
8. **Ultralytics**: Library for implementing YOLO models.  

---

## Annotation Process  
Annotation is a critical step for ensuring the model is trained on accurately labeled data. For this project:  
1. Annotate each image or video frame using Roboflow.  
2. Label individuals based on the following categories:  
   - **Helmet Only**: Annotate persons wearing only helmets.  
   - **Jacket Only**: Annotate persons wearing only safety jackets.  
   - **Both Helmet and Jacket**: Annotate persons with both safety items.  
   - **No Helmet or Jacket**: Annotate persons with neither safety item.  
3. Export the dataset in the YOLOv11-compatible format for training.

---

## Steps  

### 1. Access Kubeflow  
- Log in to Kubeflow at "http://'KUBEFLOW_IP_address':'application_port'" using your credentials on Firefox.  

### 2. Set Up Notebook Server  
- Start a new notebook server with the following configuration:  
  - **Image**: PyTorch Image.  
  - **CPU**: 8.  
  - **RAM**: 16 GB.  
  - **GPU**: (10 GB or 20 GB as required).  
  - **Persistent Storage**: None.  

### 3. Install Dependencies  
- Start a terminal in Kubeflow and execute:  
  ```bash
  cd /workspace/
  pip install ultralytics
  apt update
  apt install ffmpeg libsm6 libxext6 -y
  ```

### 4. Inference Using YOLOv11  
- Test YOLOv11’s inference capabilities:  
  ```bash
  yolo task=detect mode=predict model=yolo11n.pt source="testimg.jpg"
  ```  
- If encountering a NumPy error:  
  ```bash
  pip install numpy==1.23.4
  ```  
- For CPU-specific inference:  
  ```bash
  yolo task=detect mode=predict model=yolo11n.pt source="testimg.jpg" device='cpu'
  ```

### 5. Dataset Preparation  
- Export the annotated dataset from Roboflow:  
  - Resize images to **640x640**.  
  - Select YOLOv11 format and download using the provided code.  

- Download the dataset to the Kubeflow notebook:  
  ```bash
  mkdir custom_dataset
  cd custom_dataset/
  curl -L "<download_url>" > roboflow.zip
  unzip roboflow.zip
  rm roboflow.zip
  cd ..
  ```

### 6. Update Dataset Configuration  
- Modify `data.yaml` file under `custom_dataset/`:  
  ```yaml
  train: /workspace/custom_dataset/train/images
  val: /workspace/custom_dataset/valid/images
  test: /workspace/custom_dataset/test/images
  ```
- Rename the file to `custom_data.yaml`.  

### 7. Train the Model  
- Execute the training command:  
  ```bash
  yolo task=detect mode=train model=yolo11n.pt data="custom_dataset/custom_data.yaml" epochs=10 imgsz=640
  ```  
- If training is slow due to CPU usage:  
  - Check CUDA version:  
    ```bash
    nvcc --version
    ```  
  - Match PyTorch and CUDA versions, then reinstall PyTorch:  
    ```bash
    pip uninstall torch torchvision torchaudio
    pip install torch==<version> torchvision==<version> torchaudio==<version> -f https://download.pytorch.org/whl/torch_stable.html
    ```  

### 8. Evaluate Results  
- Training results are saved in `runs/detect/train`.  
- Use the best-trained model (`best.pt`) for inference:  
  ```bash
  cp runs/detect/train/weights/best.pt /workspace
  yolo task=detect mode=predict model=best.pt source="vid.mp4"
  ```
- Results are available in `runs/detect/predict`.  

---

## Conclusion  
This project successfully demonstrated the ability to detect and classify individuals based on helmet and jacket usage. The custom YOLOv11 model, trained on a high-quality annotated dataset, achieved robust performance.  

### Learnings  
1. Deep understanding of object detection pipelines.  
2. Effective use of Kubeflow for streamlined AI workflows.  
3. Importance of detailed annotations for real-world applications.  
4. Hands-on experience with YOLOv11 and its advanced detection capabilities.  

This project contributes to workplace safety by automating safety compliance monitoring, showcasing the power of AI in practical scenarios.
