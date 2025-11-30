# implement dataset and dataloader

import numpy as np

class dataset:

    def __init__(self, name, data, label):
        self.name = name
        self.data = np.array([x.flatten() / 255 for x in data]) # normalize
        self.label = np.array(label)
        
    def __getitem__(self, index):
        """
            return and numpy array size 1x784
        """

        return self.data[index], self.label[index]
        
    def __len__(self):
        if self.data is not None:
            return len(self.data)
        return 0
    
class dataloader:

    def __init__(self, dataset, batch_size = 8, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index = np.arange(len(dataset))

    def __iter__(self):

        if self.shuffle:
            np.random.shuffle(self.index)

        for start_id in range(0, len(self.dataset), self.batch_size):
            end_id = min(len(self.dataset), start_id + self.batch_size)

            batch_indexes = self.index[start_id:end_id]

            X_batch = self.dataset.data[batch_indexes]
            y_batch = self.dataset.label[batch_indexes]

            yield X_batch, y_batch

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))
