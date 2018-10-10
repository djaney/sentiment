
K = 0.01


def build_model(data):
    model = {}

    for sentence, labels in data:
        for word in sentence.split(" "):
            if word:
                if word not in model.keys():
                    model[word] = {}
                for label in labels.keys():
                    if label not in model[word].keys():
                        model[word][label] = labels[label]
                    else:
                        model[word][label] += labels[label]

    return model


def predict(model, data):
    prediction = {}
    for word in data.split(" "):
        if word in model.keys():
            for field in model[word].keys():
                if field not in prediction.keys():
                    prediction[field] = model[word][field] + K
                else:
                    prediction[field] *= model[word][field] + K
    return prediction

def main():
    pass
