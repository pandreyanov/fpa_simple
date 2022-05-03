import numpy as np
import pandas as pd

class Model:
    ''' This is my first module'''
    def __init__(self, data, auction_ids, bids):
        self.data = data
        self.auction_ids = auction_ids
        self.bids = bids
        
    def count_bidders_by_auction_id(self):
        self.data.sort_values(by = self.auction_ids, inplace = True)
        
        self.data['__ones'] = 1
        self.data['_bidders'] = self.data.groupby(by = self.auction_ids)['__ones'].transform(sum)
        self.data.drop(columns = ['__ones'], inplace = True)
        
        frec = self.data['_bidders'].value_counts().values
        frec = frec/np.sum(frec)
        n_bids = self.data['_bidders'].value_counts().index.values
        self.frec = {int(i):j for i,j in zip(n_bids, frec)}
        