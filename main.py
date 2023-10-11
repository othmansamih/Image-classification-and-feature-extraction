import os
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

img2vec = Img2Vec()

# storing the data into a dictionary
data = {}
data_dir = "data"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
for j, dir in enumerate([train_dir, val_dir]):
    features = []
    labels = []
    for category in os.listdir(os.path.join(dir)):
        for image in os.listdir(os.path.join(dir, category)):
            img_path = os.path.join(dir, category, image)
            img = Image.open(img_path).convert("RGB")

            img_features = img2vec.get_vec(img)
            features.append(img_features)
            labels.append(category)

    data[["training data", "validation data"][j]] = features
    data[["training labels", "validation labels"][j]] = labels

# defining the model
model = RandomForestClassifier(n_estimators=10, random_state=0)

# training the model
model.fit(data["training data"], data["training labels"])

# evaluating the performance of the model
y_pred = model.predict(data["validation data"])
score = accuracy_score(data["validation labels"], y_pred)
print("The evaluation score is: ", score)

# saving the model
joblib.dump(model, "saved model.pkl")