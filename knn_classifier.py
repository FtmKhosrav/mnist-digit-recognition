from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import fetch_openml

print("Loading MNIST dataset...")

mnist = fetch_openml("mnist_784", version=1)

X = mnist.data
y = mnist.target.astype(int)

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

# Normalize pixel values

X_train = X_train / 255.0
X_test = X_test / 255.0

print("Training KNN model...")

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

print("Making predictions...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

