
import schnetpack as spk
from ase.io import read
from schnetpack.representation.painn import PaiNN
from schnetpack.interfaces.ase_interface import AtomsConverter

# export PYTHONPATH=/home/tang/opt/schnetpack/my-github-schnet/src
print(spk.__path__)

n_atom_basis = 128
n_interactions = 3
n_rbf = 20
cutoff = 5.0

atoms = read('../testdata/md_ethanol.xyz')


# calculates pairwise distances between atoms
pairwise_distance = spk.atomistic.PairwiseDistances() 
radial_basis = spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)

# define the representation
rep = PaiNN(
    n_atom_basis=n_atom_basis,
    n_interactions=n_interactions,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff)
)

converter = AtomsConverter(spk.transform.MatScipyNeighborList(cutoff))
model_inputs = converter(atoms)
model_inputs = pairwise_distance(model_inputs)

painn_features = rep(model_inputs)