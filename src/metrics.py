from sklearn.metrics import confusion_matrix, accuracy_score

def generate_confusion_matrix(true_labels, predictions, log_callback=None):
    cm = confusion_matrix(true_labels, predictions)
    acc = accuracy_score(true_labels, predictions)
    
    if log_callback:
        log_callback("Matriz de Confusão:")
        log_callback(str(cm))  
        log_callback(f"Acurácia: {acc:.2f}")
    else:
        print("Matriz de Confusão:")
        print(cm)
        print(f"Acurácia: {acc:.2f}")