import json
import re

# Load the data from the text file
with open('/remote-home/peachilk/codebase/Qwen2.5-VL/run/cassava_diagnosis_results_deepseek_v3_1epoch.json', 'r') as file:
    data_str = file.read()
    
# Parse the JSON data
data = json.loads(data_str)

# Function to extract the true label from image path
def extract_true_label(image_path):
    # Using regex to find 'cbb' or 'cmd' in the image path
    match = re.search(r'train-(cbb|cmd)-\d+\.jpg', image_path.lower())
    if match:
        return match.group(1).upper()
    return "OTHER"  # If no match is found

# Calculate accuracy
total_samples = len(data)
correct_predictions = 0
confusion_matrix = {
    'CBB': {'CBB': 0, 'CMD': 0, 'Other': 0, 'Healthy': 0},
    'CMD': {'CBB': 0, 'CMD': 0, 'Other': 0, 'Healthy': 0},
    'OTHER': {'CBB': 0, 'CMD': 0, 'Other': 0, 'Healthy': 0}
}

for item in data:
    image_path = item['image_path']
    prediction = item['prediction']
    
    true_label = extract_true_label(image_path)
    
    # Update confusion matrix
    if true_label in confusion_matrix and prediction in confusion_matrix[true_label]:
        confusion_matrix[true_label][prediction] += 1
    
    # Check if prediction matches true label
    if (true_label == 'CBB' and prediction == 'CBB') or \
       (true_label == 'CMD' and prediction == 'CMD') or \
       (true_label != 'CBB' and true_label != 'CMD' and prediction not in ['CBB', 'CMD']):
        correct_predictions += 1

# Calculate overall accuracy
accuracy = (correct_predictions / total_samples) * 100

# Print results
print(f"Total samples: {total_samples}")
print(f"Correct predictions: {correct_predictions}")
print(f"Overall accuracy: {accuracy:.2f}%")
print("\nConfusion Matrix:")
print("True Label | Predicted as CBB | Predicted as CMD | Predicted as Other | Predicted as Healthy")
print("-" * 80)

for true_label, predictions in confusion_matrix.items():
    row = f"{true_label.ljust(10)} | "
    row += f"{predictions['CBB']:<15} | "
    row += f"{predictions['CMD']:<15} | "
    row += f"{predictions['Other']:<17} | "
    row += f"{predictions['Healthy']:<19}"
    print(row)

# Calculate per-class metrics
print("\nPer-Class Metrics:")
for true_label in confusion_matrix:
    true_positives = 0
    total_class = sum(confusion_matrix[true_label].values())
    
    if true_label == 'CBB':
        true_positives = confusion_matrix[true_label]['CBB']
    elif true_label == 'CMD':
        true_positives = confusion_matrix[true_label]['CMD']
    elif true_label == 'OTHER':
        true_positives = confusion_matrix[true_label]['Other'] + confusion_matrix[true_label]['Healthy']
    
    class_accuracy = (true_positives / total_class) * 100 if total_class > 0 else 0
    print(f"{true_label} accuracy: {class_accuracy:.2f}% ({true_positives}/{total_class})")