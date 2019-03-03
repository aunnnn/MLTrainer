import torch
from torch.utils import data
import os

'''
this implementation includes
- encoding for <start> and <end> tokens
- customized chunks that starts with <start> and ends with <end>
'''

class PA4Dataset_v2(data.DataLoader):
    
    def __init__(self, encoded_file, character_index, chunk_size=100):        
        self.character_index = character_index
        # index_character dictionary for decoding tensor purposes
        self.encoded_file = encoded_file
        self.chunk_size = chunk_size
        self.text_chunks, self.reset_flags = PA4Dataset_v2.__load_text_chunks(encoded_file, chunk_size)
        self.num_chunks = len(self.text_chunks)

    def __len__(self):
        return self.num_chunks
    
    def __getitem__(self, index):            
        '''
        return tensor shape:
        - input_tensors: [chunk_size, num_features]
        - label_tensors: [chunk_size, num_features] (input left shift by 1, 
        append the first character from next chunk if necessary)
        - reset_flag: boolean, signal the current chunk is the end of a song
        '''
        cur_chunk = self.text_chunks[index]
        cur_reset_flag = self.reset_flags[index]
        
        input_tensors = torch.zeros(self.chunk_size, len(self.character_index))
        for i in range(len(cur_chunk)):
            # one hot encode input sequence tensor
            input_tensors[i, self.character_index[cur_chunk[i]]] = 1

        # label is input tensor left shift by 1, teaching forcing chunk
        label_tensors = input_tensors[1:, :]
        # reach the end of the song
        next_char_encoding = torch.zeros(1, len(self.character_index))

        if cur_reset_flag or index == self.num_chunks - 1:
            label_tensors = torch.cat([label_tensors, next_char_encoding], dim=0)
        else:
            next_char_encoding[0, self.character_index[self.text_chunks[index + 1][0]]] = 1
            label_tensors = torch.cat([label_tensors, next_char_encoding], dim=0)

        return (input_tensors, label_tensors, cur_reset_flag)

    @staticmethod
    def __load_text_chunks(encoded_file, chunk_size):
        """
        Returns a list of strings each with length of chunk_size
        """
        i = 0
        text_chunks, reset_flags = [], []
        while i < len(encoded_file):
            line = encoded_file[i: i+chunk_size]
            # chunk ends early with "`\n" as a signal for end of the character
            if "`" in line and line.index('`') < chunk_size - 1:
                text_chunks.append(encoded_file[i: i + line.index('`') + 2])
                # signal the reset of the hidden layer at the end of the chunk training
                # reset flags also signal the dataloader to pad 0 at the end of the sequence
                reset_flags.append(True)
                i = i + line.index('`') + 2
                continue
            text_chunks.append(encoded_file[i:i+chunk_size])
            reset_flags.append(False)
            i += chunk_size
        return text_chunks, reset_flags

def build_character_index(filename):
    """
    Build character index from file.
    Return a dict of {char -> index}

    Renewed:
    - ability to encode <start> and <end> as separate tokens 
    (by replacing them with <start> -> % and <end> -> `)
    """
    try:
        text_blob = encoding_start_end_tokens(filename)
        chars = set(text_blob)
        # Sort so that it's interpretable
        print('encoding <start> as \% and <end> as \`')
        return {char: i for i, char in enumerate(sorted(chars))}
    except Exception as e:
        print("Can't read file")

def encoding_start_end_tokens(filename):
    with open(filename) as f:
        text_blob = ""
        for line in f:
            if line.startswith("<start>"):
                line = "%\n"
            elif line.startswith("<end>"):
                line = "`\n"
            text_blob += line
        return text_blob
    raise RuntimeError("Can't read file")

def decode_tensors(index_char, input_tensors, label_tensors, reset_flag, chunk_size = 100):
    if len(input_tensors.shape) > 2:
        input_tensors = input_tensors.reshape(chunk_size, -1)
        label_tensors = label_tensors.reshape(chunk_size, -1)
    input_indices = input_tensors.argmax(dim=1).tolist()
    label_indices = label_tensors.argmax(dim=1).tolist()
    decode_input = "".join([index_char[i] for i in input_indices])
    decode_label = "".join([index_char[i] for i in label_indices])
    return decode_input, decode_label, reset_flag.tolist()[0]
    
def build_all_loaders(data_parent_dir, chunk_size=100, customize_loader_params=dict()):
    """
    Get all DataLoader for train, test, val.
    """
    
    # Build char index from training data
    char_index = build_character_index(os.path.join(data_parent_dir, 'train.txt'))
    index_char = {char_index[char]: char for char in char_index.keys()}
    
    all_files = ['train', 'val', 'test']    
    all_datasets = {name: PA4Dataset_v2(encoding_start_end_tokens(os.path.join(data_parent_dir, "{0}.txt".format(name))), char_index, chunk_size=chunk_size) 
                    for name in all_files}
    
    default_params = {
        'batch_size': 1, 
        'num_workers': 1,
        'pin_memory': True
    }
    params = {**default_params, **customize_loader_params}
    
    all_loaders = {name: data.DataLoader(dataset, **params) for name, dataset in all_datasets.items()}
    return all_loaders, char_index, index_char