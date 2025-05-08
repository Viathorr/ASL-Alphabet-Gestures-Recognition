import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def get_predictions(model, dataloader):
    """
    Gets the predictions from a model on a given dataloader.

    Args:
        model (nn.Module): The model to get predictions from.
        dataloader (DataLoader): The DataLoader to use for prediction.

    Returns:
        tuple: The true labels and predicted labels as numpy arrays.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for images, landmarks, labels in dataloader:
            images, landmarks, labels = images.float().cpu(), landmarks.float().cpu(), labels.cpu()

            y_pred_logits = model(images, landmarks)
            y_preds = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)

            all_preds.extend(y_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def evaluate_model(model, dataloader, class_names=None):
    """
    Evaluates a model on a given DataLoader.

    Prints out the accuracy, F1 score (macro), precision (macro), and recall (macro) of the model on the given DataLoader.

    If class_names is given, also prints a classification report.

    Returns the predicted labels and true labels as numpy arrays.
    """
    all_preds, all_labels = get_predictions(model, dataloader)

    print("\n🔍 Evaluation Metrics:")
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("F1 Score (macro):", f1_score(all_labels, all_preds, average='macro'))
    print("Precision (macro):", precision_score(all_labels, all_preds, average='macro'))
    print("Recall (macro):", recall_score(all_labels, all_preds, average='macro'))

    if class_names:
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))

    return all_preds, all_labels


