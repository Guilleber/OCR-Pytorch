datasets = {'synthtext+mjsynth': ['../../datasets/ocr/mjsynth/',
                                  '/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/synthtext/'],
            'funsd': ['/srv/storage/synalp@talc-data2.nancy.grid5000.fr/gleberre/datasets/ocr/funsd/'],
            'svt': ['../../datasets/ocr/svt/']}

models = {'satrn-large': {'d_model': 512,
                          'd_hidden': 2048,
                          'dropout': 0.1,
                          'nlayers_encoder': 12,
                          'nlayers_decoder': 6,
                          'nhead': 8}}
