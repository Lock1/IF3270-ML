from generate_model import generate_model
import numpy as np

model = str(input("Masukkan model yang ingin digunakan [relu / sigmoid]: "))
ffnn = generate_model(filename=f"{model}.json")
prediction = ffnn.predict(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
ffnn.info()
