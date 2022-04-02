import json
import jsonpickle

def saveModel(obj, filename):
    json_obj = jsonpickle.encode(obj)
    with open(f'models/{filename}.json', 'w', encoding='utf-8') as f:
        json.dump(json_obj, f, ensure_ascii=True, indent=4)
    print(f'Model saved to {filename}.json')

def loadModel(filename):
    with open(f'models/{filename}.json') as json_file:
        data = json.load(json_file)

    python_class = jsonpickle.decode(data)

    return python_class