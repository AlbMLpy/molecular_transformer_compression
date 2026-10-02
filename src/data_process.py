"""
Code based on:
- https://github.com/ardigen/MAT/tree/master
- "Edge Attention-based Multi-Relational Graph Convolutional Networks" ->
    -> https://github.com/Luckick/EAGCN
- "Convolutional Embedding of Attributed Molecular Graphs for Physical 
    Property Prediction" -> https://github.com/connorcoley/conv_qsar_fast
"""

import os
import pickle
import logging

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.metrics import pairwise_distances

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor


def load_data_from_df(
    dataset_path: str, 
    add_dummy_node: bool = True, 
    one_hot_formal_charge: bool = False, 
    use_data_saving: bool = True,
    fingerprints: bool = False,
    radius: int = 2,
) -> tuple[list, list] | tuple[np.ndarray, np.ndarray]:
    """Load and featurize molecular data from a CSV file.

    The input CSV is expected to contain two columns: SMILES strings and
    corresponding target labels. Depending on ``fingerprints``, the
    function either generates MAT molecular graph features or Morgan
    fingerprints.

    Parameters
    ----------
    dataset_path : str
        Path to the CSV file containing the molecular data. The first column
        must contain SMILES strings and the second column must contain the
        corresponding target labels.
    add_dummy_node : bool, optional
        If ``True``, add a dummy node to each molecular graph. This is used
        by MAT to support molecular-level aggregation. Defaults to ``True``.
    one_hot_formal_charge : bool, optional
        If ``True``, encode atomic formal charges using one-hot encoding.
        Otherwise, formal charge is stored as a numerical feature.
        Defaults to ``False``.
    use_data_saving : bool, optional
        If ``True``, load previously computed features from the dataset
        directory when available. If no cached features are found, compute
        the features and save them for subsequent runs. Defaults to ``True``.
    fingerprints : bool, optional
        If ``True``, generate Morgan fingerprints instead of MAT graph
        features. Defaults to ``False``.
    radius : int, optional
        Radius of the local molecular environment used to generate Morgan
        fingerprints. Only used when ``fingerprints=True``. Defaults to ``2``.

    Returns
    -------
    tuple[list, list] or tuple[np.ndarray, np.ndarray]
        If ``fingerprints=False``, returns ``(X, y)`` where ``X`` is a list
        of molecular graph features. Each molecule is represented by a tuple
        of node features, adjacency matrix, and distance matrix, and ``y``
        contains the corresponding labels.

        If ``fingerprints=True``, returns ``(X, y)`` as NumPy arrays, where
        ``X`` contains the Morgan fingerprint vectors and ``y`` contains the
        corresponding labels.
    """
    feat_stamp = (
        f'{"_dn" if add_dummy_node else ""}' 
        + f'{"_ohfc" if one_hot_formal_charge else ""}'
    )
    feat_stamp = feat_stamp + "_fingerprints" if fingerprints else feat_stamp
    feature_path = dataset_path.replace('.csv', f'{feat_stamp}.p')
    
    if use_data_saving and os.path.exists(feature_path):
        logging.info(f"Loading features stored at '{feature_path}'")
        x_all, y_all = pickle.load(open(feature_path, "rb"))
        return x_all, y_all

    data_df = pd.read_csv(dataset_path)
    data_x, data_y = data_df.iloc[:, 0].values, data_df.iloc[:, 1].values

    if data_y.dtype == np.float64:
        data_y = data_y.astype(np.float32)

    if fingerprints:
        x_all = np.stack(
            [smiles_to_morgan(smiles, radius=radius) for smiles in data_x]
        )
        y_all = np.asarray(data_y)
    else:
        x_all, y_all = load_data_from_smiles(
            data_x, 
            data_y, 
            add_dummy_node=add_dummy_node,
            one_hot_formal_charge=one_hot_formal_charge,
        )

    if use_data_saving and not os.path.exists(feature_path):
        logging.info(f"Saving features at '{feature_path}'")
        pickle.dump((x_all, y_all), open(feature_path, "wb"))

    return x_all, y_all


def load_data_from_smiles(
    x_smiles: list[str], 
    labels: list[float], 
    add_dummy_node: bool = True, 
    one_hot_formal_charge: bool = False,
) -> tuple[list, list]:
    """Load and featurize molecular data from SMILES strings and labels.

    Each SMILES string is converted into a molecular graph representation
    containing atom features, an adjacency matrix, and a distance matrix.
    Molecules that cannot be parsed or featurized are skipped and a warning
    is logged.

    Parameters
    ----------
    x_smiles : list[str]
        List of SMILES strings representing the molecules.
    labels : list[float]
        List of target values corresponding to the SMILES strings. The
        ordering must match ``x_smiles``.
    add_dummy_node : bool, optional
        If ``True``, add a dummy node to each molecular graph. This is used
        by MAT for molecular-level representation and aggregation.
        Defaults to ``True``.
    one_hot_formal_charge : bool, optional
        If ``True``, encode atomic formal charges using one-hot encoding.
        Otherwise, formal charge is stored as a numerical feature.
        Defaults to ``False``.

    Returns
    -------
    tuple[list, list]
        A tuple ``(X, y)`` where ``X`` is a list of molecular graph
        representations. Each molecule is represented by:

        - node features,
        - adjacency matrix,
        - pairwise distance matrix.

        ``y`` is a list containing the corresponding target values.
    """
    x_all, y_all = [], []
    for smiles, label in zip(x_smiles, labels):
        try:
            mol = MolFromSmiles(smiles)
            try:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, maxAttempts=5000)
                AllChem.UFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
            except:
                AllChem.Compute2DCoords(mol)

            afm, adj, dist = featurize_mol(
                mol, add_dummy_node, one_hot_formal_charge
            )
            x_all.append([afm, adj, dist])
            y_all.append([label])
        except ValueError as e:
            msg_str = (
                f'the SMILES ({smiles}) can not be '
                + f'converted to a graph.\nREASON: {e}'
            )
            logging.warning(msg_str)
    return x_all, y_all


def smiles_to_morgan(smiles: str, radius: int = 2) -> np.ndarray:
    """Compute Morgan fingerprint features for a SMILES string."""
    mol = MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=2048)
    return np.asarray(fp)


def featurize_mol(
    mol, 
    add_dummy_node: bool, 
    one_hot_formal_charge: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an RDKit molecule into MAT molecular graph features.

    The molecular graph representation consists of atom-level features,
    a binary adjacency matrix describing molecular connectivity, and a
    pairwise distance matrix computed from the molecule's 3D conformer.

    Parameters
    ----------
    mol : rdchem.Mol
        RDKit molecule to featurize. The molecule is expected to have a
        conformer containing atomic 3D coordinates.
    add_dummy_node : bool
        If ``True``, add a dummy node to the molecular graph. The dummy
        node is used by MAT for molecular-level representation and
        aggregation.
    one_hot_formal_charge : bool
        If ``True``, encode atomic formal charges using one-hot encoding.
        Otherwise, formal charge is stored as a numerical feature.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:

        - ``node_features`` : np.ndarray
            Atom-level feature matrix with shape
            ``(n_atoms, n_features)``.
        - ``adjacency_matrix`` : np.ndarray
            Molecular connectivity matrix with shape
            ``(n_atoms, n_atoms)``.
        - ``distance_matrix`` : np.ndarray
            Pairwise Euclidean distance matrix derived from the 3D
            atomic coordinates, with shape ``(n_atoms, n_atoms)``.

        If ``add_dummy_node=True``, each matrix includes the additional
        dummy node.
    """
    node_features = np.array(
        [
            get_atom_features(atom, one_hot_formal_charge) 
            for atom in mol.GetAtoms()
        ]
    )

    adj_matrix = np.eye(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

    conf = mol.GetConformer()
    pos_matrix = np.array(
        [
            [
                conf.GetAtomPosition(k).x, 
                conf.GetAtomPosition(k).y, 
                conf.GetAtomPosition(k).z,
            ]
            for k in range(mol.GetNumAtoms())
        ]
    )
    dist_matrix = pairwise_distances(pos_matrix)

    if add_dummy_node:
        m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
        m[1:, 1:] = node_features
        m[0, 0] = 1.
        node_features = m

        m = np.zeros((adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1))
        m[1:, 1:] = adj_matrix
        adj_matrix = m

        m = np.full((dist_matrix.shape[0] + 1, dist_matrix.shape[1] + 1), 1e6)
        m[1:, 1:] = dist_matrix
        dist_matrix = m

    return node_features, adj_matrix, dist_matrix


def get_atom_features(atom, one_hot_formal_charge: bool = True) -> np.ndarray:
    """Extract atom-level features from an RDKit atom.

    The feature vector encodes the atom's chemical properties, including
    atomic number, number of neighboring atoms, total number of hydrogens,
    formal charge, ring membership, and aromaticity.

    Parameters
    ----------
    atom : rdchem.Atom
        RDKit atom from which the features are extracted.
    one_hot_formal_charge : bool, optional
        If ``True``, encode the formal charge using one-hot encoding.
        Otherwise, store the formal charge as a numerical value.
        Defaults to ``True``.

    Returns
    -------
    np.ndarray
        One-dimensional NumPy array containing the atom-level features.
        The feature vector is stored with ``float32`` precision.
    """
    attributes = []
    attributes += one_hot_vector(
        atom.GetAtomicNum(),
        [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
    )
    attributes += one_hot_vector(
        len(atom.GetNeighbors()),
        [0, 1, 2, 3, 4, 5]
    )
    attributes += one_hot_vector(
        atom.GetTotalNumHs(),
        [0, 1, 2, 3, 4]
    )

    if one_hot_formal_charge:
        attributes += one_hot_vector(
            atom.GetFormalCharge(),
            [-1, 0, 1]
        )
    else:
        attributes.append(atom.GetFormalCharge())

    attributes.append(atom.IsInRing())
    attributes.append(atom.GetIsAromatic())

    return np.array(attributes, dtype=np.float32)


def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)


class Molecule:
    """Container for a single molecular graph and its target value.

    Stores the molecular features required by MAT together with the
    corresponding target label and dataset index.

    Parameters
    ----------
    x : tuple[np.ndarray, np.ndarray, np.ndarray]
        Molecular graph representation containing node features,
        adjacency matrix, and pairwise distance matrix.
    y : float or np.ndarray
        Target value associated with the molecule.
    index : int
        Original index of the molecule in the dataset.

    Attributes
    ----------
    node_features : np.ndarray
        Atom-level feature matrix.
    adjacency_matrix : np.ndarray
        Molecular connectivity matrix.
    distance_matrix : np.ndarray
        Pairwise distance matrix derived from the molecular conformer.
    y : float or np.ndarray
        Target value for the molecule.
    index : int
        Original dataset index of the molecule.
    """

    def __init__(self, x, y, index):
        self.node_features = x[0]
        self.adjacency_matrix = x[1]
        self.distance_matrix = x[2]
        self.y = y
        self.index = index


class MolDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable
    for PyTorch. Note that this class inherits torch.utils.data.Dataset.
    """

    """PyTorch dataset for storing and accessing molecular graph data.

    This dataset wraps a list of :class:`Molecule` objects and provides
    the interface required by ``torch.utils.data.Dataset``. It supports
    both individual molecule access and slicing.

    Parameters
    ----------
    data_list : list[Molecule]
        List of ``Molecule`` objects containing the molecular features
        and corresponding target values.

    Notes
    -----
    Indexing with an integer returns a single ``Molecule`` object.
    Slicing returns a new ``MolDataset`` containing the selected molecules.
    """

    def __init__(self, data_list: list[Molecule]):
        self.data_list = data_list

    def __len__(self):
        """Return the number of molecules in the dataset."""
        return len(self.data_list)

    def __getitem__(self, key):
        """Return a molecule or a subset of the dataset.

        Parameters
        ----------
        key : int or slice
            Integer index for a single molecule or slice for a subset.

        Returns
        -------
        Molecule or MolDataset
            A single ``Molecule`` for integer indexing, or a new
            ``MolDataset`` for slicing.
        """
        if type(key) == slice:
            return MolDataset(self.data_list[key])
        return self.data_list[key]


def pad_array(
    array: np.ndarray, 
    shape: tuple[int, int], 
    dtype=np.float32
) -> np.ndarray:
    """Pad a two-dimensional array with zeros to a specified shape.

    Parameters
    ----------
    array : np.ndarray
        Two-dimensional array to be padded. Its dimensions must not exceed
        the corresponding dimensions of ``shape``.
    shape : tuple[int, int]
        Desired shape of the output array.
    dtype : data-type, optional
        Data type of the padded array. Defaults to ``np.float32``.

    Returns
    -------
    np.ndarray
        Zero-padded array with the specified shape and data type.

    Raises
    ------
    ValueError
        If ``array`` has more than two dimensions or its shape is larger
        than the requested output shape.
    """
    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2-dimensional array, got {array.ndim} dimensions."
        )

    if array.shape[0] > shape[0] or array.shape[1] > shape[1]:
        raise ValueError(
            f"Array shape {array.shape} cannot be padded to {shape}."
        )
    
    padded_array = np.zeros(shape, dtype=dtype)
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array


def mol_collate_func(batch: list[Molecule]) -> list:
    """Collate molecules into a padded batch for MAT.

    Molecules may contain different numbers of atoms, so their node
    features, adjacency matrices, and distance matrices are padded to the
    maximum number of atoms in the batch. The resulting arrays are
    converted to PyTorch tensors.

    Parameters
    ----------
    batch : list[Molecule]
        List of :class:`Molecule` objects to collate into a batch.

    Returns
    -------
    list[torch.FloatTensor]
        A list containing four tensors in the following order:

        - ``adjacency_matrix`` : Padded molecular adjacency matrices.
        - ``node_features`` : Padded atom-level feature matrices.
        - ``distance_matrix`` : Padded pairwise distance matrices.
        - ``labels`` : Target values for each molecule in the batch.

        The first three tensors are padded along their atom dimensions to
        the maximum number of atoms in the batch.
    """
    adjacency_list, distance_list, features_list, labels = [], [], [], []
    max_size = 0
    for molecule in batch:
        if type(molecule.y[0]) == np.ndarray:
            labels.append(molecule.y[0])
        else:
            labels.append(molecule.y)
        if molecule.adjacency_matrix.shape[0] > max_size:
            max_size = molecule.adjacency_matrix.shape[0]

    for molecule in batch:
        adjacency_list.append(
            pad_array(molecule.adjacency_matrix, (max_size, max_size))
        )
        distance_list.append(
            pad_array(molecule.distance_matrix, (max_size, max_size))
        )
        features_list.append(
            pad_array(
                molecule.node_features, 
                (max_size, molecule.node_features.shape[1])
            )
        )
    return [
        FloatTensor(features) 
        for features in (adjacency_list, features_list, distance_list, labels)
    ]


def construct_dataset(x_all, y_all) -> MolDataset:
    """Construct a molecular dataset from features and target values.

    Each pair of molecular features and target values is wrapped in a
    :class:`Molecule` object and assigned its corresponding index in the
    input data.

    Parameters
    ----------
    x_all : list
        List of molecular graph representations. Each element should
        contain node features, an adjacency matrix, and a distance matrix.
    y_all : list
        List of target values corresponding to the molecular features in
        ``x_all``. The ordering must match ``x_all``.

    Returns
    -------
    MolDataset
        Dataset containing the constructed :class:`Molecule` objects.

    Notes
    -----
    The inputs ``x_all`` and ``y_all`` should have matching lengths.
    """
    output = [
        Molecule(data[0], data[1], i)
        for i, data in enumerate(zip(x_all, y_all))
    ]
    return MolDataset(output)


def construct_loader(
    x: list, 
    y: list, 
    batch_size: int, 
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """Construct a DataLoader for molecular graph data.

    The input molecular features and target values are first wrapped in a
    :class:`MolDataset`. Batches are collated using :func:`mol_collate_func`,
    which pads molecules with different numbers of atoms to a common size.

    Parameters
    ----------
    x : list
        List of molecular graph representations. Each element should
        contain node features, an adjacency matrix, and a distance matrix.
    y : list
        List of target values corresponding to the molecules in ``x``.
        The ordering must match ``x``.
    batch_size : int
        Number of molecules to include in each batch.
    shuffle : bool, optional
        If ``True``, shuffle the dataset before creating batches.
        Defaults to ``True``.

    Returns
    -------
    torch.utils.data.DataLoader
        Data loader that yields padded batches of molecular features and
        corresponding target values.
    """
    data_set = construct_dataset(x, y)
    return torch.utils.data.DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        collate_fn=mol_collate_func,
        shuffle=shuffle,
    )


def random_split_indices(
    n_samples: int,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random train/validation/test splits of dataset indices.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset.
    train_ratio : float
        Fraction assigned to training.
    valid_ratio : float
        Fraction assigned to validation.
    test_ratio : float
        Fraction assigned to testing.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_idx : np.ndarray
    valid_idx : np.ndarray
    test_idx : np.ndarray
    """
    if not np.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + valid_ratio + test_ratio must be 1.0")

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    rng = np.random.RandomState(seed)

    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_train = int(n_samples * train_ratio)
    n_valid = int(n_samples * valid_ratio)

    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]
    return train_idx, valid_idx, test_idx


def load_split_data(
    data_path: str, 
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_seed: int = 42, 
    fingerprints: bool = False,
) -> tuple[tuple, tuple, tuple]:
    X, y = load_data_from_df(data_path, fingerprints=fingerprints)
    train_idx, valid_idx, test_idx = random_split_indices(
        len(X), train_ratio, valid_ratio, test_ratio, split_seed,
    )

    if fingerprints:
        return (
            (X[train_idx], y[train_idx]),
            (X[valid_idx], y[valid_idx]),
            (X[test_idx], y[test_idx]),
        )
    else:
        return (
            ([X[i] for i in train_idx], [y[i] for i in train_idx]),
            ([X[i] for i in valid_idx], [y[i] for i in valid_idx]),
            ([X[i] for i in test_idx], [y[i] for i in test_idx]),
        )
