from sklearn.metrics import confusion_matrix, accuracy_score

def generate_confusion_matrix(true_labels, predictions):
    cm = confusion_matrix(true_labels, predictions)
    acc = accuracy_score(true_labels, predictions)
    print("Matriz de Confusão:")
    print(cm)
    print(f"Acurácia: {acc:.2f}")
