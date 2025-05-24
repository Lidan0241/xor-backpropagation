- pour générer un dataset
`python generate_xor_dataset.py -n 100 -p 0.2 -o xor_dataset/xor_20.csv`

- pour entraîner
`python mlp_train.py config.json xor_dataset/xor_20.csv --seed 42`