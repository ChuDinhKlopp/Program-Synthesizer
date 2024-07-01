import json
import random

with open("manimml-data-7k.json") as f:
    data = json.load(f)

random.shuffle(data)

train_data = data[:int((len(data)+1)*0.80)]
val_data = data[int((len(data)+1)*0.80):]

with open("train.json", "w") as out:
    json.dump(train_data, out, indent=4)

with open("val.json", "w") as out:
    json.dump(val_data, out, indent=4)
