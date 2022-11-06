from tqdm import tqdm
import torch
import requests
import numpy as np
from PIL import Image

if __name__ == '__main__':
    from transformers import CLIPModel, CLIPProcessor
    from pybooru import Danbooru
    clip_name = 'openai/clip-vit-base-patch32'

    clipprocessor = CLIPProcessor.from_pretrained(clip_name)
    if torch.cuda.is_available():
        clipmodel = CLIPModel.from_pretrained(clip_name).to('cuda').eval()
    else:
        clipmodel = CLIPModel.from_pretrained(clip_name).to('cpu').eval()

    print(f'loaded {clip_name}')

    client = Danbooru('danbooru', username='haru1367', api_key='')

use_cuda = torch.cuda.is_available()

def text_embeddings(text, model, processor):
    inputs = processor(text=text, return_tensors='pt', padding=True)['input_ids']
    if use_cuda:
        inputs = inputs.to('cuda')
    result = model.get_text_features(input_ids=inputs).cpu().detach().numpy()
    return (result / np.linalg.norm(result, axis=1, keepdims=True)).squeeze(axis=0)

def image_embeddings(url, model, processor):
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    inputs = processor(images=image, return_tensors='pt')['pixel_values']
    if use_cuda:
        inputs = inputs.to('cuda')
    result = model.get_image_features(pixel_values=inputs).cpu().detach().numpy()
    return (result / np.linalg.norm(result)).squeeze(axis=0)

def image_embeddings_file(path, model, processor):
    image = Image.open(path).convert('RGB')
    inputs = processor(images=image, return_tensors='pt')['pixel_values']
    if use_cuda:
        inputs = inputs.to('cuda')
    result = model.get_image_features(pixel_values=inputs).cpu().detach().numpy()
    return (result / np.linalg.norm(result)).squeeze(axis=0)

import glob
import json

@torch.inference_mode()
def directory_embed(directory, model=None, processor=None):
    image_paths = glob.glob(f'{directory}/*')
    post_dict = {}
    with open('aesthetic.json') as f:
        post_dict = json.load(f)
    with open(f"{directory.split('/')[-1]}.npy", 'wb') as fp:
        embs = []
        for i in tqdm(image_paths):
            embs.append((image_embeddings_file(i, model, processor), post_dict[i.split('/')[-1].split('.')[0]]))
        np.save(fp, embs)
        with open(f"{directory.split('/')[-1]}-avgemb.npy", 'wb') as fpavg:
            avg_emb = np.mean(embs, axis=0)
            np.save(fpavg, avg_emb)


# binary classifier that consumes CLIP embeddings
class Classifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = torch.nn.Linear(hidden_size//2, output_size)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    

# train the classifier
def train_classifier(train_data, train_labels, model):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    epochs = 100
    # make dataloader
    train_data = torch.from_numpy(train_data).float()
    train_labels = torch.from_numpy(train_labels).float()
    # train_data is a numpy array of embeddings
    # train_labels is a numpy array of labels that are 0 or 1
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
    
    total_steps = len(train_loader) * epochs
    # setup tqdm
    pbar = tqdm(range(total_steps), desc='Total Steps', leave=False)


    # move everything to cuda if available
    if use_cuda:
        model = model.to('cuda')

    loss_value = 0.0
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.to('cuda'))
            loss = criterion(outputs.squeeze(1), labels.to('cuda'))
            loss.backward()
            optimizer.step()
            # report to tqdm
            pbar.update(1)
            logs = {'loss': loss.detach().item(), 'epoch': epoch}
            pbar.set_postfix(logs)
            loss_value = loss.detach().item()

    print(f'Finished Training, loss: {loss_value}')

    return model

if __name__ == '__main__':
    # train it.

    #directory_embed('aesthetic', clipmodel, clipprocessor)

    train_data = np.load('aesthetic.npy', allow_pickle=True)
    # each element in train_data is a tuple of (embedding, bool)
    train_dataset = []
    train_labels = []
    for i in train_data:
        train_labels.append(i[1])
        train_dataset.append(i[0])

    model = Classifier(512, 256, 1)
    model = train_classifier(np.array(train_dataset), np.array(train_labels), model)

    try:
        while True:
            file_url = input('Enter a file path or url: ')
            # get the embedding
            image_embeds = image_embeddings(file_url, clipmodel, clipprocessor)
            # get the prediction
            prediction = model(torch.from_numpy(image_embeds).float().to('cuda'))
            print(f'Prediction: {prediction.item()}')
    except KeyboardInterrupt:
        print('Exiting...')
    except requests.exceptions.MissingSchema:
        print('Invalid URL')
        pass

    torch.save(model.state_dict(), 'aesthetic.pth')
    # load it back in
    model = Classifier(512, 256, 1)
    model.load_state_dict(torch.load('aesthetic.pth'))
