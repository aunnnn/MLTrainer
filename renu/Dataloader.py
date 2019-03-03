import torch

unique_chars = {'m': 0, '/': 1, '{': 2, '!': 3, 'X': 4, '4': 5, 'x': 6, 'N': 7, '>': 8, 'F': 9, 'q': 10, 'U': 11, 'j': 12, 'W': 13, '-': 14, 'b': 15, '(': 16, 'K': 17, 'A': 18, 'T': 19, 'O': 20, 'M': 21, '.': 22, ')': 23, 'c': 24, 'a': 25, ']': 26, '@': 27, '9': 28, 'C': 29, 'v': 30, '2': 31, '#': 32, '*': 33, 'B': 34, 'n': 35, 'R': 36, '|': 37, '\t': 38, 'H': 39, '}': 40, '1': 41, 'p': 42, 'D': 43, '<': 44, 'I': 45, 's': 46, 'J': 47, 'G': 48, '8': 49, '3': 50, 'i': 51, '~': 52, '\\': 53, 'o': 54, 'E': 55, '"': 56, 'y': 57, '[': 58, 't': 59, '6': 60, '?': 61, 'P': 62, ':': 63, '_': 64, 'S': 65, ' ': 66, ',': 67, 'l': 68, 'd': 69, '0': 70, 'k': 71, 'Q': 72, "'": 73, 'Y': 74, 'g': 75, '&': 76, 'L': 77, '5': 78, 'u': 79, '\n': 80, 'h': 81, '=': 82, 'z': 83, 'Z': 84, '^': 85, '+': 86, 'f': 87, 'V': 88, 'r': 89, 'w': 90, '7': 91, 'e': 92}

class Dataloader():
    def __init__(self, dictionary=unique_chars):
        self.num_chars=len(dictionary)
        self.dictionary=dictionary
    
    def __hotencode_char(self,char):
        output = torch.zeros(self.num_chars)
        index = self.dictionary[char]
        output[index] = 1
        return output
    
    def encode_string(self,line):
        hotencoded_seq=[]
        for char in line:
            hotencoded_seq.append(self.__hotencode_char(char))
        return torch.stack(hotencoded_seq,dim=0)
  
    
    def __encode_file(self,filepath):
        hotencoded_seq, numerized_seq = [],[]

        f = open(filepath, "r")
        line = f.readline()
        while line:
            for char in line:
                hotencoded_seq.append(self.__hotencode_char(char))
                numerized_seq.append(self.dictionary[char])
            line = f.readline()
            
        f.close()
        
        return torch.stack(hotencoded_seq,dim=0), torch.tensor(numerized_seq)
   
    def load_data(self,filepath, chunk_size=100):
        '''
        public function: returns encoded sequence from text file
        returns a list of chunks
        each chunk is a list of hot encoded tensors
        '''
        hotencoded_sequence, numerized_sequence = self.__encode_file(filepath)
        inputs,labels = hotencoded_sequence[:-1], numerized_sequence[1:]
            
        if chunk_size:
            input_chunks=[]
            label_chunks=[]

            i=0
            while i < inputs.size()[0]:
                input_chunks.append(inputs[i:(i+chunk_size)])
                label_chunks.append(labels[i:(i+chunk_size)])
                i+=chunk_size

            return input_chunks,label_chunks
        else:
            return inputs,labels
  
    
    