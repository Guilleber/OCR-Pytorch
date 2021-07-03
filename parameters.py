datasets = {'synthtext+mjsynth': {'train': ['../../datasets/ocr/mjsynth/train.jsonl',
                                               '/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/synthtext/train.jsonl'],
                                  'val': ['../../datasets/ocr/svt/train.jsonl'],
                                  'test': ['../../datasets/ocr/svt/test.jsonl']}}

models = {'satrn-large': {'d_model': 512,
                          'd_hidden': 2048,
                          'dropout': 0.1,
                          'nlayers_encoder': 12,
                          'nlayers_decoder': 6,
                          'nhead': 8,
                          'positional_enc': 'a2dpe'},
          'satrn-large-exp': {'d_model': 512,
                              'd_hidden': 2048,
                              'dropout': 0.1,
                              'nlayers_encoder': 12,
                              'nlayers_decoder': 6,
                              'nhead': 8,
                              'positional_enc': 'experimental'}}
