import os
import requests
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
import numpy as np

from protein import ProteinState


def fetch_protein(pdb_id: str) -> ProteinState:
    # pull protein from PDB
    pdb_file = f"{pdb_id}.pdb"
    pdb_file_path = os.path.join(os.getcwd(), pdb_file)
    protein_url = f"https://files.rcsb.org/download/{pdb_file}"
    req = requests.get(protein_url)
    with open(pdb_file_path, "w") as f:
        f.write(req.text)

    # parse PDB data for phi/psi angles
    structure = PDBParser().get_structure(pdb_id, pdb_file)
    peptides = PPBuilder().build_peptides(structure)[0]
    phi_psi_angles = list(map(lambda x: (180.0 if not x[0] else (x[0] * 180 / np.pi) % 360,
                                         180.0 if not x[1] else (x[1] * 180 / np.pi) % 360),
                              peptides.get_phi_psi_list()))

    protein_state = ProteinState(angles=np.array(phi_psi_angles).T)
    return protein_state
