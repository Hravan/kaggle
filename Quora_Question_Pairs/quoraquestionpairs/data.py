from torchtext import data

def get_dataset(file: str, question: data.Field):
    '''
    Consume a path to a file and a question representation and return a dataset.
    '''
    dataset = data.TabularDataset(path=file,
                                  format='csv',
                                  fields=[('id', None),
                                  ('qid1', None),
                                  ('qid2', None),
                                  ('question1', question),
                                  ('question2', question),
                                  ('is_duplicate', data.Field(sequential=False,
                                                              use_vocab=False,
                                                              is_target=True))],
                                  skip_header=True)
    return dataset
