import joblib
from PIL import Image
from img2vec_pytorch import Img2Vec

# loading the model
model = joblib.load("saved model.pkl")

# loading the image
img_path = input("Enter the path of the image: ")
img = Image.open(img_path).convert("RGB")

# making an inference (prediction)
img2vec = Img2Vec()
features = img2vec.get_vec(img)
y_pred = model.predict([features])
print(y_pred)