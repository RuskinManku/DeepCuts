import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
class STSBDL(Dataset):
    def __init__(self, hug_data):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.hug_data=hug_data

    def __len__(self):
        return len(self.hug_data)

    def __getitem__(self, idx):
        return ((self.hug_data[idx]['sentence1'],self.hug_data[idx]['sentence2']),self.hug_data[idx]['similarity_score'])

from transformers import BertTokenizer
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
dataset=load_dataset("stsb_multi_mt", name="en", split="train")
dataset_psudo_dl=STSBDL(dataset)
dataset_dl=DataLoader(dataset_psudo_dl,batch_size=3)
for x,y in dataset_dl:
    list_x=x[0]
    list_x_2=x[1]
    tokenized=tokenizer(list_x,list_x_2,padding=True,return_tensors="pt")
    print(x)
    print(y)
    print(tokenizer.decode(tokenized.input_ids[0]))
    exit()