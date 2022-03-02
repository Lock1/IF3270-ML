from generate_model import generate_model
import numpy as np

ffnn = generate_model(filename='model.json')
prediction = ffnn.predict(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
ffnn.info()
