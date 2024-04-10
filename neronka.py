import numpy as np
import json
import shutil
import numpy as np

from settings import *

def load_neronka(files):
    vesa = {}
    with open(files) as f:
        file = json.load(f)
        for i in file:
            vesa[i] = np.asarray(file[i])
    return vesa

def save_neronka(files, vesa):
    #files без расширения файла
    shutil.copyfile(f'{files}.json', f'{files}1.json')
    with open(f'{files}.json', 'w') as f:
        json.dump(vesa, f)

def load_dataset():
    with np.load("mnist.npz") as f:
        # convert from RGB to Unit RGB
        x_train = f['x_train'].astype("float32") / 255

        # reshape from (60000, 28, 28) into (60000, 784)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

        # labels
        y_train = f['y_train']

        # convert to output layer format
        y_train = np.eye(10)[y_train]

        return x_train, y_train

if __name__ == '__main__':
    images, labels = load_dataset()
    weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
    weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))
    bias_input_to_hidden = np.zeros((20, 1))
    bias_hidden_to_output = np.zeros((10, 1))

    epochs = 10
    e_loss = 0
    e_correct = 0
    learning_rate = 0.01

    for epoch in range(epochs):
        print(f"Epoch №{epoch}")

        for image, label in zip(images, labels):
            image = np.reshape(image, (-1, 1))
            label = np.reshape(label, (-1, 1))


            hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
            hidden = 1 / (1 + np.exp(-hidden_raw)) # sigmoid


            output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
            output = 1 / (1 + np.exp(-output_raw))


            e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
            e_correct += int(np.argmax(output) == np.argmax(label))


            delta_output = output - label
            weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
            bias_hidden_to_output += -learning_rate * delta_output


            delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
            weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
            bias_input_to_hidden += -learning_rate * delta_hidden


        save_neronka('vesa_ner', {'weights_input_to_hidden': weights_input_to_hidden.tolist(), 'weights_hidden_to_output': weights_hidden_to_output.tolist(),
        'bias_input_to_hidden': bias_input_to_hidden.tolist(), 'bias_hidden_to_output': bias_hidden_to_output.tolist()})

        print(f"Потерь: {round((e_loss[0] / images.shape[0]) * 100, 3)}%")
        print(f"Точность: {round((e_correct / images.shape[0]) * 100, 3)}%")
        e_loss = 0
        e_correct = 0
    

