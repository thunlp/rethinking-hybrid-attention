import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--acts_file", type=str, required=True, help="path to .npz file")
parser.add_argument("--model_alias", type=str, default="model")
parser.add_argument("--classifier", type=str, default="logistic", 
                    choices=["logistic", "mlp", "svm", "random_forest", "gradient_boosting", "naive_bayes", "knn"],
                    help="Classifier type: logistic (default), mlp, svm, random_forest, gradient_boosting, naive_bayes, knn")
parser.add_argument(
    "--feature_type",
    type=str,
    default="residual",
    choices=["residual", "delta"],
    help="Feature type to probe: residual uses saved layer hidden states, delta uses per-layer additions h_l - h_{l-1}",
)
args = parser.parse_args()

def create_classifier(classifier_type, random_state=42):
    """Create a classifier based on the specified type.
    
    Args:
        classifier_type: str, one of ['logistic', 'mlp', 'svm', 'random_forest', 
                                      'gradient_boosting', 'naive_bayes', 'knn']
        random_state: int, random seed for reproducibility
    
    Returns:
        sklearn classifier object
    """
    if classifier_type == "logistic":
        return LogisticRegression(
            random_state=random_state, 
            max_iter=1000, 
            solver='liblinear', 
            C=0.1, 
            multi_class='ovr'
        )
    elif classifier_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=10000,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
    elif classifier_type == "svm":
        return SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,  # Enable probability estimates
            random_state=random_state,
            max_iter=1000
        )
    elif classifier_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1
        )
    elif classifier_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state
        )
    elif classifier_type == "naive_bayes":
        return GaussianNB()
    elif classifier_type == "knn":
        return KNeighborsClassifier(
            n_neighbors=5,
            weights='uniform',
            algorithm='auto'
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

def get_layer_keys(npz_file, feature_type):
    """Get layer feature keys from the npz file for the requested feature type."""
    key_prefix = "layer_" if feature_type == "residual" else "delta_layer_"
    layer_keys = [k for k in npz_file.files if k.startswith(key_prefix)]
    layer_keys.sort(key=lambda x: int(x.split("_")[-1]))
    return layer_keys

def extract_layer_idx(layer_name):
    """Extract numeric layer index from keys like layer_3 or delta_layer_3."""
    return int(layer_name.split("_")[-1])

def main():
    print(f"Loading activations from {args.acts_file}...")
    print(f"Model alias: {args.model_alias}")
    
    data = np.load(args.acts_file)
    labels = data["labels"]
    
    layer_keys = get_layer_keys(data, args.feature_type)
    if not layer_keys:
        available_keys = [k for k in data.files if "layer" in k]
        raise ValueError(
            f"No {args.feature_type} features found in {args.acts_file}. "
            f"Available layer-like keys: {available_keys}"
        )
    
    accuracies = []
    layers = []
    print(f"Using classifier: {args.classifier}")
    print(f"Using feature type: {args.feature_type}")
    print(f"Training probes for {len(layer_keys)} layers...")
    print(f"Class count: {len(np.unique(labels))}, chance baseline: {1.0 / max(len(np.unique(labels)), 1):.4f}")
    
    for layer_name in tqdm(layer_keys):
        X = data[layer_name] # [num_samples, hidden_dim]
        y = labels
        
        stratify_y = y if len(np.unique(y)) > 1 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=stratify_y,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=None,
            )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        clf = create_classifier(args.classifier, random_state=42)
        clf.fit(X_train, y_train)
        
        acc = clf.score(X_test, y_test)
        accuracies.append(acc)
        layer_idx = extract_layer_idx(layer_name)
        layers.append(layer_idx)

    print("\nLayer-wise accuracy:")
    for l, acc in zip(layers, accuracies):
        print(f"Layer {l}: {acc:.4f}")

    best_idx = int(np.argmax(accuracies))
    print(f"\nBest layer: {layers[best_idx]} (accuracy={accuracies[best_idx]:.4f})")

if __name__ == "__main__":
    main()