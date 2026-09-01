# PlantVillage Dataset Statistics

This document contains reference statistics and information about the PlantVillage dataset used for training the CropSage model, based on the Exploratory Data Analysis (EDA) and data splitting phases.

## 📊 Overall Summary

* **Total Images (Scanned by EDA):** 108,606 (Across all variants)
* **Total Images Used (Color Variant):** 54,305 
* **Number of Classes:** 38
* **Minimum Images in a Class:** 304
* **Maximum Images in a Class:** 11,014
* **Mean Images per Class:** ~2,858
* **Class Imbalance Ratio (Max/Min):** 36.23
* **Corrupted Images:** 0

## ✂️ Data Split (80/10/10 Stratified)

* **Training Set:** 43,444 images (80%)
* **Validation Set:** 5,430 images (10%)
* **Test Set:** 5,431 images (10%)

## 📂 Class Labels (38 Classes)

The dataset contains 38 distinct crop-disease combinations. A weight array is applied during PyTorch training to handle the class imbalance.

1. `Apple___Apple_scab`
2. `Apple___Black_rot`
3. `Apple___Cedar_apple_rust`
4. `Apple___healthy`
5. `Blueberry___healthy`
6. `Cherry_(including_sour)___Powdery_mildew`
7. `Cherry_(including_sour)___healthy`
8. `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot`
9. `Corn_(maize)___Common_rust_`
10. `Corn_(maize)___Northern_Leaf_Blight`
11. `Corn_(maize)___healthy`
12. `Grape___Black_rot`
13. `Grape___Esca_(Black_Measles)`
14. `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)`
15. `Grape___healthy`
16. `Orange___Haunglongbing_(Citrus_greening)`
17. `Peach___Bacterial_spot`
18. `Peach___healthy`
19. `Pepper,_bell___Bacterial_spot`
20. `Pepper,_bell___healthy`
21. `Potato___Early_blight`
22. `Potato___Late_blight`
23. `Potato___healthy`
24. `Raspberry___healthy`
25. `Soybean___healthy`
26. `Squash___Powdery_mildew`
27. `Strawberry___Leaf_scorch`
28. `Strawberry___healthy`
29. `Tomato___Bacterial_spot`
30. `Tomato___Early_blight`
31. `Tomato___Late_blight`
32. `Tomato___Leaf_Mold`
33. `Tomato___Septoria_leaf_spot`
34. `Tomato___Spider_mites Two-spotted_spider_mite`
35. `Tomato___Target_Spot`
36. `Tomato___Tomato_Yellow_Leaf_Curl_Virus`
37. `Tomato___Tomato_mosaic_virus`
38. `Tomato___healthy`

## 📈 Outputs Generated
* **Class Distribution Chart:** `notebooks/outputs/class_distribution.png`
* **Sample Images Grid:** `notebooks/outputs/sample_images_grid.png`
* **Class Indices Mapping:** `notebooks/outputs/class_indices.json`
* **Class Weights:** `notebooks/outputs/class_weights.json`
