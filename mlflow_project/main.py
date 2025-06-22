from utils.data_loader import load_data
from experiments.decision_tree import run as run_decision_tree
from experiments.random_forest import run as run_random_forest

# Load the dataset
X_train, X_test, y_train, y_test = load_data()

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

if __name__ == "__main__":
    print("Running Decision Tree...")
    run_decision_tree(X_train, X_test, y_train, y_test)

    print("Running Random Forest...")
    run_random_forest(X_train, X_test, y_train, y_test)
