datasets = {'synthtext+mjsynth': {'train': ['/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/mjsynth/train.jsonl',
                                            '/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/mjsynth/val.jsonl',
                                            '/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/mjsynth/test.jsonl',
                                            '/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/synthtext/all.jsonl'],
                                  'val': ['../../datasets/ocr/svt/train.jsonl',
                                          '/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/IIIT5K/train.jsonl',
                                          '/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/ICDAR13/train.jsonl',
                                          '/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/ICDAR15/train.jsonl'],
                                  'test': ['../../datasets/ocr/svt/test.jsonl',
                                          '/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/IIIT5K/test.jsonl',
                                          '/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/ICDAR13/test.jsonl',
                                          '/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/ICDAR15/test.jsonl']},
            'iam': {
                'train': ['/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/iam/train_words.jsonl'],
                'val': ['/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/iam/val_words.jsonl'],
                'test': ['/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/iam/test_words.jsonl']
            },
            'iam+emnist': {
                'train': ['/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/iam/train.jsonl',
                    '/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/emnist/train.jsonl'],
                'val': ['/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/iam/val.jsonl'],
                'test': ['/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/iam/test.jsonl']
            },
            'funsd': {
                'train': ['/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/docbank/train_small.jsonl'],
                'val': ['/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/funsd/train.jsonl'],
                'test': ['/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/funsd/test.jsonl']
            }}


models = {'satrn-large': {'d_model': 512,
                          'd_hidden': 2048,
                          'dropout': 0.1,
                          'nlayers_encoder': 12,
                          'nlayers_decoder': 6,
                          'nhead': 8,
                          'positional_enc': 'simple'},
          'satrn-large-exp': {'d_model': 512,
                              'd_hidden': 2048,
                              'dropout': 0.1,
                              'nlayers_encoder': 12,
                              'nlayers_decoder': 6,
                              'nhead': 8,
                              'positional_enc': 'experimental'}}
