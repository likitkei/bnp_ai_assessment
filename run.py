import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import glob
import re
import os
from difflib import SequenceMatcher
import crnn as crnn
import generateTextImage

# Generate Text Image for prediction
generateTextImage.generate()

# Sort the images by their file names
def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

model_path = './pre_trained_model/crnn.pth'
img_dir = './data/*.jpg'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

print('\n\n')
print('Predicting text images......')
model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load(model_path)) # Load the pre-trained model parameters to the current model

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
images = glob.glob(img_dir)
images.sort(key=natural_keys)
pred_result = []
for image in images:
    with open(image, 'rb') as file:
        img = Image.open(file).convert('L')
        img = transformer(img)
        if torch.cuda.is_available():
            img = img.cuda()
        img = img.view(1, *img.size())
        img = Variable(img)

        model.eval()
        preds = model(img)
        _, preds = preds.max(2)
        preds = preds.view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        pred_result.append(sim_pred)
        print('%s: %-20s => %-20s' % (image, raw_pred, sim_pred))

        with open(os.path.join('./data', "prediction_words.txt"), 'w', encoding="utf8") as f:
            for i in range(len(pred_result)):
                # file_name = str(i) + "." + args.extension
                f.write("{}\n".format(pred_result[i]))

print('------------------------Summary for mispredicted words--------------------------- \n')
with open('./data/prediction_words.txt', 'r') as f1:
    prediction = f1.read().split('\n')

with open('./data/true_words.txt', 'r') as f2:
    original = f2.read().split('\n')

matched = []
different = []
difference_count = []
small, large = (prediction, original) if len(prediction) <= len(original) else (original, prediction)
position = -1

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

for x, y in zip(small, large):
    if x == y:
        matched.append(y)
        position = position + 1
    else:
        position = position + 1
        print('True word :', x)
        print('Predicted word :', y)
        print('Similarity: ', similarity(x,y))
        print('------------------------------------------------------------------ \n')
        difference_count.append(sum(1 for a, b in zip(x, y) if a != b) + abs(len(x) - len(y)))
        different.append(y)
print('Total number of mispredicted words: {} out of {}' .format(len(difference_count), len(original) - 1))