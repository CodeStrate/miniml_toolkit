import numpy as np

def r2_score(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    y_test_bar = np.mean(y_test)
    residual_squared_error_sum = np.sum((y_test - y_pred)**2)
    total_squared_dev_sum = np.sum((y_test - y_test_bar)**2)
    return np.round(1 - (residual_squared_error_sum/total_squared_dev_sum), 3)

def mean_squared_error(y_pred: np.ndarray, y_test: np.ndarray, squared: bool = True) -> float:
    if squared:
        return np.round(np.sum((y_test - y_pred)**2) / len(y_test), 3)
    else:
        return np.round(np.sqrt(np.sum((y_test - y_pred)**2) / len(y_test)), 3)
    
class ConfusionMatrix:
    def __init__(self, y_pred: np.ndarray, y_true: np.ndarray):
        self.y_pred = y_pred
        self.y_true = y_true
        self.cm = None
    
    def _make_cnf_matrix(self):
        labels = np.unique(self.y_true)
        self.cm = np.zeros((len(labels), len(labels)), dtype=int)

        for true, pred in zip(self.y_true, self.y_pred):
            self.cm[true, pred] += 1

    def show(self):
        return self.cm

    def clf_report(self):
        """
        Generate a classification report with precision, recall, F1-score, and support.
        Returns a dictionary with metrics for each class and overall metrics.
        """
        if self.cm is None:
            self._make_cnf_matrix()
        
        n_classes = self.cm.shape[0]
        
        # Initialize metrics
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        f1_score = np.zeros(n_classes)
        support = np.zeros(n_classes)
        
        # Calculate metrics for each class
        for i in range(n_classes):
            # True Positives, False Positives, False Negatives
            tp = self.cm[i, i]
            fp = np.sum(self.cm[:, i]) - tp
            fn = np.sum(self.cm[i, :]) - tp
            
            # Support (actual number of samples for this class)
            support[i] = np.sum(self.cm[i, :])
            
            # Precision = TP / (TP + FP)
            if tp + fp > 0:
                precision[i] = tp / (tp + fp)
            else:
                precision[i] = 0.0
            
            # Recall = TP / (TP + FN)
            if tp + fn > 0:
                recall[i] = tp / (tp + fn)
            else:
                recall[i] = 0.0
            
            # F1-score = 2 * (precision * recall) / (precision + recall)
            if precision[i] + recall[i] > 0:
                f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            else:
                f1_score[i] = 0.0
        
        # Overall accuracy
        accuracy = np.trace(self.cm) / np.sum(self.cm)
        
        # Macro averages (unweighted mean)
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1_score)
        
        # Weighted averages (weighted by support)
        total_support = np.sum(support)
        weighted_precision = np.sum(precision * support) / total_support if total_support > 0 else 0.0
        weighted_recall = np.sum(recall * support) / total_support if total_support > 0 else 0.0
        weighted_f1 = np.sum(f1_score * support) / total_support if total_support > 0 else 0.0
        
        # Create report dictionary
        report = {
            'class_metrics': {},
            'accuracy': np.round(accuracy, 3),
            'macro_avg': {
                'precision': np.round(macro_precision, 3),
                'recall': np.round(macro_recall, 3),
                'f1-score': np.round(macro_f1, 3),
                'support': int(total_support)
            },
            'weighted_avg': {
                'precision': np.round(weighted_precision, 3),
                'recall': np.round(weighted_recall, 3),
                'f1-score': np.round(weighted_f1, 3),
                'support': int(total_support)
            }
        }
        
        # Add per-class metrics
        for i in range(n_classes):
            report['class_metrics'][i] = {
                'precision': np.round(precision[i], 3),
                'recall': np.round(recall[i], 3),
                'f1-score': np.round(f1_score[i], 3),
                'support': int(support[i])
            }
        
        return report
    
    def print_report(self):
        """
        Print a formatted classification report similar to sklearn's classification_report.
        """
        if self.cm is None:
            self._make_cnf_matrix()
        
        report = self.clf_report()
        
        print("\nClassification Report")
        print("=" * 60)
        print(f"{'Class':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 60)
        
        # Print per-class metrics
        for class_id, metrics in report['class_metrics'].items():
            print(f"{class_id:<8} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                  f"{metrics['f1-score']:<10.3f} {metrics['support']:<10}")
        
        print("-" * 60)
        
        # Print overall metrics
        print(f"{'Accuracy':<8} {'':<10} {'':<10} {report['accuracy']:<10.3f} "
              f"{report['macro_avg']['support']:<10}")
        print(f"{'Macro Avg':<8} {report['macro_avg']['precision']:<10.3f} "
              f"{report['macro_avg']['recall']:<10.3f} {report['macro_avg']['f1-score']:<10.3f} "
              f"{report['macro_avg']['support']:<10}")
        print(f"{'Weighted Avg':<8} {report['weighted_avg']['precision']:<10.3f} "
              f"{report['weighted_avg']['recall']:<10.3f} {report['weighted_avg']['f1-score']:<10.3f} "
              f"{report['weighted_avg']['support']:<10}")
        print("=" * 60)