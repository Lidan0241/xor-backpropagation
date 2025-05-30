# xor-backpropagation – XOR avec bruit, ReLU et analyse de complexité

Ce projet met en œuvre un perceptron multicouche (MLP) de type **2-2-1** pour apprendre la fonction booléenne XOR. Il inclut une analyse complète des performances avec et sans bruit, une comparaison de fonctions d’activation (sigmoïde vs ReLU bornée), ainsi qu’une étude de complexité en temps et en espace.

- Génération d'un dataset XOR bruité
`python3 generate_xor_dataset.py -n 100 -p 0.2 -o xor_dataset/xor_20.csv`

- `-n` : nombre d'exemples
- `-p` : proportion de bruit (entre 0 et 1)
- `-o` : nom de fichier de sortie

- Entraînement du MLP
`python3 mlp_train.py config.json xor_dataset/xor_20.csv --seed 42`

Les scripts suivants mesurent le temps moyen et la mémoire allouée :

- `measure_single_neuron.py` : pour un seul neurone
- `measure_hidden_layer.py` : pour toute la couche cachée
- `measure_full_network.py` : pour le réseau complet
- `measure_full_network_relu.py` : pour le réseau complet avec ReLU comme activation

Afin d'étudier l'effet du choix de la fonction d’activation sur les performances du MLP, nous avons modifié le fichier `mlp_train.py` pour créer une nouvelle version nommée `mlp_train_relu.py`. L’activation est définie dans `activation.py`.
