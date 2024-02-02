import os
import cv2
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import seaborn as sns


def load_and_preprocess_test_data(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(folder_path, filename))
            resized_img = cv2.resize(img, (50, 50))  # Resize image to 50x50
            flattened_img = resized_img.ravel()  # Flatten the image
            images.append(flattened_img)
            labels.append(label)
    return images, labels

model = joblib.load('/Users/rohithsiddi/Desktop/untitled folder/face_recognition_model.pkl')

test_folder_path_class1 = '/Users/rohithsiddi/Desktop/untitled folder/test_images/Rohith_120'
test_folder_path_class2 = '/Users/rohithsiddi/Desktop/untitled folder/test_images/Susmitha_100'

test_images_class1, test_labels_class1 = load_and_preprocess_test_data(test_folder_path_class1, 'Rohith_120')
test_images_class2, test_labels_class2 = load_and_preprocess_test_data(test_folder_path_class2, 'Susmitha_100')

test_images = test_images_class1 + test_images_class2
test_labels = test_labels_class1 + test_labels_class2

test_features = np.array(test_images)
predicted_labels = model.predict(test_features)

conf_matrix = confusion_matrix(test_labels, predicted_labels)
print("Confusion Matrix:\n", conf_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=['Class 1', 'Class 2'], yticklabels=['Class 1', 'Class 2'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

class_report = classification_report(test_labels, predicted_labels)
print("Classification Report:\n", class_report)

file_path = "/Users/rohithsiddi/Desktop/untitled folder/classification_report.txt"  # Update with your desired path

# Write the report to a file
with open(file_path, "w") as file:
    file.write(class_report)

print(f"Classification report saved to {file_path}")
from sklearn.preprocessing import label_binarize

# Binarize labels
# If the output is 1D, convert it to 2D
binarized_labels = label_binarize(test_labels, classes=['Rohith_120', 'Susmitha_100'])
if binarized_labels.ndim == 1:
    binarized_labels = binarized_labels.reshape(-1, 1)

# Assuming that the positive class is 'Susmitha_100' and it's the second column in the probabilities
positive_class_probabilities = model.predict_proba(test_features)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(binarized_labels[:, 0], positive_class_probabilities)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:0.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

