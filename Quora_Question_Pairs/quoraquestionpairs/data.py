import torchtext

def get_simple_dataset(file):
    '''Get a TabularDataset for a given file.'''
    return torchtext.data.TabularDataset(path=file,
                                         format='csv',
                                         fields=[('id', None),
                                         ('qid1', None),
                                         ('qid2', None),
                                         ('question1', question),
                                         ('question2', question),
                                         ('is_duplicate', data.Field(sequential=False,
                                                                     use_vocab=False))],
                                         skip_header=True)
