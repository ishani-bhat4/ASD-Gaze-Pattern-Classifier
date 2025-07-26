# 👁️ Classifying Gaze Patterns for Preemptive Screening of Autism Spectrum Disorder (ASD) in Children Using Deep Learning

## 📌 Abstract

Autism Spectrum Disorder (ASD) is a lifelong neurodevelopmental condition. The incidence of autism in children is rising sharply, presenting a growing challenge for healthcare systems globally. Although atypical behavior often begins to show as early as 12–18 months of age, the lack of definitive biomarkers impedes timely diagnosis and intervention.

This research investigates the viability of using gaze patterns as early biomarkers for autism by leveraging the **Saliency4ASD** dataset. Our study compares traditional machine learning models with neural network architectures and demonstrates that deep learning approaches significantly outperform conventional methods in classifying atypical gaze behavior. By enabling earlier screening, this approach holds promise for facilitating timely therapeutic interventions.

---

## ✨ Key Highlights

- 📈 Uses gaze-based saliency maps for behavioral pattern recognition
- 🧠 Implements and evaluates neural networks vs traditional ML algorithms
- 🧪 Based on the **Saliency4ASD** benchmark dataset
- 📊 Achieves higher classification accuracy with CNN architectures
- 🎯 Designed for early ASD screening in children aged 12–36 months

---

## 📚 Published Paper

**Title:**  
Classifying Gaze Patterns for Preemptive Screening of Autism Spectrum Disorder (ASD) in Children Using Deep Learning

**Authors:**  
G Dhruva, Ishani Bhat, G Sivagamasundari

**Conference:**  
2025 International Conference on Innovative Trends in Information Technology (**ICITIIT**)

**Pages:**  
1–6

**Date:**  
February 21, 2025

**Publisher:**  
IEEE

> https://www.researchgate.net/profile/Dhruva-Guruprasad/publication/393118852_Classifying_Gaze_Patterns_for_Preemptive_Screening_of_Autism_Spectrum_Disorder_ASD_in_Children_Using_Deep_Learning/links/68640409e4632b045dc8ee5d/Classifying-Gaze-Patterns-for-Preemptive-Screening-of-Autism-Spectrum-Disorder-ASD-in-Children-Using-Deep-Learning.pdf  

---

## 🖼️ Dataset Used

- **Saliency4ASD Dataset**  
  A public benchmark dataset containing eye gaze saliency maps of children with and without ASD while viewing social scenes.  
 ---

## 💡 Methodology

1. **Preprocessing** gaze-based saliency maps and resizing for CNN input.
2. **Baseline Models:** SVM, k-NN, Random Forests for binary classification.
3. **Deep Learning Models:** Convolutional Neural Networks (CNN) for visual pattern recognition.
4. **Training/Evaluation:** Split data using stratified 5-fold cross-validation and report accuracy, precision, recall, and F1-score.
5. **Result:** CNN-based model achieved significant improvement in classification accuracy over traditional approaches.

---

## 🧪 Results

| Model              | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| SVM               | 76.5%    | 75.2%     | 74.8%  | 74.9%    |
| Random Forest     | 78.1%    | 77.4%     | 76.3%  | 76.8%    |
| **CNN (ours)**    | **89.3%**| **88.7%** | **90.1%**| **89.4%**|

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib
- Scikit-learn
- OpenCV

---

## 🧠 Future Work

- Incorporate **temporal gaze sequence analysis**
- Explore **multimodal inputs** (facial expressions, pupil dilation)
- Develop a **web/mobile prototype** for clinical/parental use

---

## 👤 Authors

- G Dhruva  
- **Ishani Bhat** – [GitHub](https://github.com/your-github) | [LinkedIn](https://linkedin.com/in/your-linkedin)  
- G Sivagamasundari

---

## 📃 License

This research is released under the [MIT License](LICENSE).  
For academic use only. Please cite our work if you use this in your research.

---

## 📝 Citation

```bibtex
@inproceedings{bhat2025asd,
  title={Classifying Gaze Patterns for Preemptive Screening of Autism Spectrum Disorder (ASD) in Children Using Deep Learning},
  author={G Dhruva and Ishani Bhat and G Sivagamasundari},
  booktitle={2025 International Conference on Innovative Trends in Information Technology (ICITIIT)},
  pages={1--6},
  year={2025},
  publisher={IEEE}
}
