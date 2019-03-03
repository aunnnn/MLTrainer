import torch
from torch.utils import data
import os

class PA4Dataset(data.DataLoader):
    
    def __init__(self, filename, character_index, chunk_size=100):        
        self.character_index = character_index
        self.filename = filename
        
        self.text_chunks = PA4Dataset.__load_text_chunks(filename, chunk_size)
        self.num_chunks = len(self.text_chunks)
        
    def __len__(self):
        return self.num_chunks
    
    def __getitem__(self, index):            
        i_chunk = index
        cur_chunk = self.text_chunks[i_chunk]
        
        input_tensors = torch.zeros(len(cur_chunk), len(self.character_index))
        label_tensors = torch.zeros(len(cur_chunk), len(self.character_index))
        
        for i in range(len(cur_chunk)-1):
            char = cur_chunk[i]
            next_char = cur_chunk[i+1]
            
            input_tensors[i, self.character_index[char]] = 1
            label_tensors[i, self.character_index[next_char]] = 1
                        
        # How to handle the last chunk? Currently will use '\n' to maintain continuity
        # (since last char of every music are '<end>\n' anyway).
        if i_chunk == self.num_chunks - 1:
            eof_i = len(cur_chunk) - 1
            input_tensors[eof_i, self.character_index['>']] = 1
            label_tensors[eof_i, self.character_index['\n']] = 1
        else:
            # Last char will have first char of next chunk
            last_i = len(cur_chunk) - 1
            
            cur_chunk_last_char = cur_chunk[last_i]
            next_chunk_first_char = self.text_chunks[i_chunk+1][0]
            
            input_tensors[last_i, self.character_index[cur_chunk[last_i]]] = 1
            label_tensors[last_i, self.character_index[next_chunk_first_char]] = 1
                
        return (input_tensors, label_tensors)
        
    @staticmethod
    def __load_text_chunks(filename, chunk_size):
        """
        Returns a list of strings each with length chunk_size
        """
        text_chunks = []
        with open(filename) as f:
            text_blob = f.read()
            for i in range(0, len(text_blob), chunk_size):
                text_chunks.append(text_blob[i:i+chunk_size])
            return text_chunks
        raise RuntimeError("Can't read file")

def build_character_index(filename):
    """
    Build character index from file.
    Return a dict of {char -> index}
    """
    with open(filename) as f:
        text_blob = f.read()
        chars = sorted(set(text_blob))
        # Sort so that it's interpretable
        char_2_index = {char: i for i, char in enumerate(chars)}
        index_2_char = {i: char for i, char in enumerate(chars)}
        return char_2_index, index_2_char
    
    raise RuntimeError("Can't read file")
    
def build_all_loaders(data_parent_dir, chunk_size=100, customize_loader_params=dict()):
    """
    Get all DataLoader for train, test, val.
    """
    
    # Build char index from training data
    char_2_index, index_2_char = build_character_index(os.path.join(data_parent_dir, 'train.txt'))
    
    all_files = ['train', 'val', 'test']    
    all_datasets = {name: PA4Dataset(os.path.join(data_parent_dir, "{0}.txt".format(name)), char_2_index, chunk_size=chunk_size)
                    for name in all_files}
    
    default_params = {
        'batch_size': 1, 
        'num_workers': 1,
        'pin_memory': True
    }
    params = {**default_params, **customize_loader_params}
    
    all_loaders = {name: data.DataLoader(dataset, **params) for name, dataset in all_datasets.items()}
    return all_loaders, {'char_2_index': char_2_index, 
                         'index_2_char': index_2_char}