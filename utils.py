import pandas as pd 
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import math
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from sklearn.metrics.pairwise import cosine_similarity
import os


def mol_idx_image(mol):
    """
    Setting the Atom Indexes as Map Numbers to the molecules
    
    Args:
    mol: rdkit.Chem.Mol 
    
    Returns:
    mol with its indexes set as AtomMapNumbers"""

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom.SetAtomMapNum(idx)
    return mol


def find_attachment_points(whole_mol, scaffold_mol, scaffold_smiles, show=True):
    """
    Identify and label attachment points in a molecule relative to a scaffold.

    Args:
        whole_mol (rdkit.Chem.Mol): Full molecule.
        scaffold_mol (rdkit.Chem.Mol): Scaffold molecule.
        scaffold_smiles (str): SMILES string of the scaffold.
        show (bool): If True, display the highlighted scaffold image.

    Returns:
        dict with:
            - attachment_points (list[int]): Indices of scaffold atoms that are attachment points.
            - frag_repl (dict): Map from attachment point index -> fragment SMILES.
            - selected_attachments (list[int]): User-chosen attachment points.
            - labelled_scaff (rdkit.Chem.Mol): Scaffold molecule with atom notes.
    """

    if not whole_mol.HasSubstructMatch(scaffold_mol):
        raise ValueError("Scaffold is not a substructure of the whole molecule.")

    # 1. Get substructure matches
    matches = whole_mol.GetSubstructMatches(scaffold_mol)
    reverse_map = {m: i for i, m in enumerate(matches[0])}

    attachment_points = []
    nbr_map = []

    # 2. Find atoms in whole_mol whose neighbors are not in scaffold
    for whole_idx in reverse_map.keys():
        whole_atom = whole_mol.GetAtomWithIdx(whole_idx)
        neighbors = whole_atom.GetNeighbors()
        for nbr in neighbors:
            if nbr.GetIdx() not in reverse_map:
                nbr_map.append((reverse_map[whole_idx], nbr.GetIdx()))
                attachment_points.append(reverse_map[whole_idx])

    attachment_points = sorted(set(attachment_points))

    # 3. Label scaffold atoms
    labelled_scaff = Chem.MolFromSmiles(scaffold_smiles)
    for atts in attachment_points:
        label_atom = labelled_scaff.GetAtomWithIdx(atts)
        label_atom.SetProp("atomNote", str(atts))

    # 4. Show image
    if show:
        img = Draw.MolToImage(
            labelled_scaff,
            size=(500, 500),
            highlightAtoms=attachment_points,
            highlightColor=(0, 0, 1)
        )
        display(img)

    # 5. Get fragments after removing scaffold atoms
    e_whfrags = Chem.EditableMol(whole_mol)
    for frag_idx in sorted(matches[0], reverse=True):
        e_whfrags.RemoveAtom(frag_idx)
    remainder = e_whfrags.GetMol()
    wh_rem = Chem.GetMolFrags(remainder, asMols=True, sanitizeFrags=True)

    frag_repl = {}
    for f in wh_rem:
        atom_indices = [
            a.GetIntProp("molAtomMapNumber") if a.HasProp("molAtomMapNumber") else a.GetIdx()
            for a in f.GetAtoms()
        ]
        for atts in nbr_map:
            if atts[1] in atom_indices:
                frag_repl[atts[0]] = Chem.MolToSmiles(f)

    # 6. User selection
    chosen_attachments = input(f"Select attachment points from {attachment_points}, comma-separated: ")
    selected_attachments = []
    for sel in chosen_attachments.split(","):
        if sel.strip().isdigit():
            val = int(sel.strip())
            if val in attachment_points:
                selected_attachments.append(val)
            else:
                print(f"{val} not present in the presented attachment points")

    frag_repl = {att:values for att,values in frag_repl.items() if att in selected_attachments}


    return {
        "attachment_points": attachment_points,
        "frag_repl": frag_repl,
        "selected_attachments": selected_attachments,
        "labelled_scaff": labelled_scaff
    }


def calculate_cutoff(fragment_df, selected_attachments, revised_total_mols=None):
    """
    Calculate total molecules possible and cutoff for descriptors.

    Parameters
    ----------
    fragment_df : pandas.DataFrame
        DataFrame containing fragment information.
    selected_attachments : list
        List of selected attachment points.
    revised_total_mols : int, optional
        User-defined number of molecules to generate. 
        If None, function will prompt the user.

    Returns
    -------
    total_molecules : int
        Maximum number of molecules possible.
    revised_total_mols : int
        Final number of molecules user wants to generate.
    desc_cutoff : int
        Descriptor cutoff value.
    """
    total_molecules = fragment_df.shape[0] ** len(selected_attachments)
    print(f"Total possible molecules: {total_molecules}")

    if revised_total_mols is None:
        revised_total_mols = int(input(f"Mention the total number of molecules you wish to generate. "
                                       f"Total possible: {total_molecules} "))
        

    desc_cutoff = math.ceil(revised_total_mols ** (1 / len(selected_attachments)))
    print("No. of molecules to generate:",revised_total_mols)
    if revised_total_mols>total_molecules:
        print("Error: The number of molecules for generation is greater than total possible")
    return total_molecules, revised_total_mols, desc_cutoff


def calc_props(smi):
    
    """
    Calculating a set of selected descriptors for a smile
     
    Args: the smile of the specified molecule or fragment

    Returns: A dict with the descriptor name as the key and the calculated descriptor as the value
       
     """
    
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES:",smi)
     
    props = {
            "MolWt": Descriptors.MolWt(mol),
            "LogP": Crippen.MolLogP(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "HBA": Lipinski.NumHAcceptors(mol),
            "HBD": Lipinski.NumHDonors(mol),
            "RotatableBonds": Lipinski.NumRotatableBonds(mol)
        }
    return props

import re
import pandas as pd
from rdkit import Chem

def process_fragments(fragment_df, calc_props_func):
    """
    Process a DataFrame of fragments by replacing attachment points with 'C',
    calculating properties, and appending them to the DataFrame.

    Parameters
    ----------
    fragment_df : pd.DataFrame
        DataFrame containing a column 'Fragments' with SMILES.
    calc_props_func : function
        A function that takes an RDKit Mol object and returns a dict of descriptors.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with:
        - 'att_repl' column (fragments with attachment points replaced by 'C')
        - descriptor columns appended
    """
    # Regex pattern for attachment points
    pattern = r"\[[^\]]*\*[^\]]*\]"

    # Replace attachment points with "C"
    fragment_df = fragment_df.copy()  # avoid modifying original
    fragment_df["att_repl"] = fragment_df["Fragments"].str.replace(pattern, "C", regex=True)

    desc_dict = []
    
    for smi in fragment_df['att_repl']:
        if smi is not None:
            props = calc_props_func(smi)
        else:
            props = {}  # fallback in case SMILES parsing fails
        desc_dict.append(props)

    # Merge descriptor DataFrame
    fragment_df = pd.concat([fragment_df, pd.DataFrame(desc_dict)], axis=1)

    return fragment_df




def process_and_save_similar_fragments(frag_repl, fragment_desc, cutoff, save_dir="Similar_frag"):

    """
    Calculate descriptor-based similarity between replacement fragments and a fragment library,
    filter the top similar fragments for each attachment point, save them as CSV files, 
    and return them in memory.

    Args:
        frag_repl (dict): A dictionary mapping attachment point identifiers (keys) to 
                          SMILES strings of fragments to be replaced (values).
        fragment_desc (pd.DataFrame): DataFrame containing fragment descriptors including 
                                      columns ['Fragments', 'att_repl', 'MolWt', 'LogP', 
                                      'TPSA', 'HBA', 'HBD', 'RotatableBonds'].
        cutoff (int): Number of top similar fragments to retain per attachment point.
        save_dir (str, optional): Directory to save the filtered CSV files. Defaults to "Similar_frag".

    Returns:
        dict: A dictionary mapping each attachment point to a DataFrame of the top similar fragments 
              (descriptor-based) for that point. Each DataFrame contains the original fragment 
              descriptors and an additional 'Similarity' column.
    
    Behavior:
        - For each attachment point, calculates similarity of the replacement fragment to all 
          fragments in `fragment_desc` based on six descriptors (MolWt, LogP, TPSA, HBA, HBD, RotatableBonds).
        - Filters the top `cutoff` fragments by similarity.
        - Saves each filtered DataFrame as `att_<att_point>.csv` in `save_dir`.
        - Returns all filtered DataFrames in memory as a dictionary.
    """
    
    os.makedirs(save_dir, exist_ok=True)

    similar_dfs = {}  # keep dataframes in memory
    # print(fragment_desc.shape[0])
    fragment_desc = fragment_desc.drop_duplicates(subset = ["att_repl"])
    # print(fragment_desc.shape[0])

    for att_point, frag_smi in frag_repl.items():
        # calculate properties of replacement fragment
        frag_props = calc_props(frag_smi)

        # similarity calculation with existing fragment_desc
        similarities = []
        for idx, row in fragment_desc.iterrows():
            sim = 0
            for prop in ['MolWt', 'LogP', 'TPSA', 'HBA', 'HBD', 'RotatableBonds']:
                sim += 1 - abs(row[prop] - frag_props[prop]) / (abs(row[prop]) + 1e-6)
            sim /= 6  # average similarity
            similarities.append(sim)

        fragment_desc["Similarity"] = similarities

        # filter top molecules
        filtered_df = fragment_desc.nlargest(cutoff, "Similarity").copy().reset_index(drop=True)
        # print("Its the new one")
        filtered_df = filtered_df.dropna(subset = 'Fragments')

        # store in memory
        similar_dfs[att_point] = filtered_df

        # also save as CSV
        csv_path = os.path.join(save_dir, f"att_{att_point}.csv")
        filtered_df.to_csv(csv_path, index=False)

    return similar_dfs

def att_to_C_conversion(mol, keep_idx, att_indices):

    """
    Convert specified atoms in a molecule to carbon, except for a designated atom to keep.

    Args:
        mol (rdkit.Chem.Mol): The input RDKit molecule.
        keep_idx (int): The index of the atom to retain (will NOT be converted to carbon).
        att_indices (list of int): List of atom indices to convert to carbon.

    Returns:
        rdkit.Chem.Mol: A new RDKit molecule where all atoms in `att_indices` (except `keep_idx`) 
                        have been converted to carbon (atomic number 6).
    """
        
    rw = Chem.RWMol(mol)
    for idx in att_indices:
        if idx != keep_idx:
            rw.GetAtomWithIdx(idx).SetAtomicNum(6)
    return rw.GetMol()

def prepare_fragments(fragment_df):

    """
    Process a fragment DataFrame to generate molecules with individual attachment points preserved 
    and other dummy atoms converted to carbon.

    Args:
        fragment_df (pd.DataFrame): A DataFrame containing at least a column named "Fragments" 
                                    with SMILES strings representing fragments. 
                                    Fragments should have dummy atoms ('*') indicating attachment points.

    Returns:
        pd.DataFrame: An expanded DataFrame with the following new columns:
            - 'mol' (rdkit.Chem.Mol): RDKit molecule object for each fragment.
            - 'dupli_count' (int): Number of dummy atoms ('*') in the fragment.
            - 'att_points' (list of int): Indices of all attachment points in the molecule.
            - 'new_mol' (rdkit.Chem.Mol): Molecule with all attachment points converted to carbon except 
                                           one preserved for each row.
            - 'frag_att' (int): The index of the attachment point preserved in 'new_mol'.
    """

    fragment_df = fragment_df.dropna(subset=["Fragments"]).reset_index(drop=True)
    fragment_df["mol"] = fragment_df["Fragments"].apply(lambda frag: Chem.MolFromSmiles(frag))
    fragment_df['dupli_count'] = fragment_df['mol'].apply(
        lambda mol: sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "*")
    )
    fragment_df = fragment_df[fragment_df['dupli_count'] > 0].reset_index(drop=True)

    fragment_df['att_points'] = fragment_df['mol'].apply(
        lambda mol: [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"]
    )

    duplicated = []
    for i, m in enumerate(fragment_df['dupli_count']):
        row = fragment_df.loc[[i]]
        duplicated.append(pd.concat([row]*m, ignore_index=True))
    expanded = pd.concat(duplicated, ignore_index=True)

    new_mols, frag_atts = [], []
    for i in range(len(expanded)):
        mol = expanded.loc[i, 'mol']
        att_indices = expanded.loc[i, 'att_points']
        dupli_idx = expanded.loc[i, 'dupli_count']
        keep_idx = att_indices[i % dupli_idx]
        new_mols.append(att_to_C_conversion(mol, keep_idx, att_indices))
        frag_atts.append(keep_idx)

    expanded['new_mol'] = new_mols
    expanded['frag_att'] = frag_atts
    return expanded

import re
import pandas as pd
from rdkit import Chem

def generate_final_molecules(combinations, scaffold_mol, similar_dfs, att_to_C_conversion):
    """
    Generate combined molecules from scaffold and fragments, clean SMILES, and return a DataFrame.
    
    Parameters
    ----------
    combinations : list
        List of fragment index combinations.
    scaffold_mol : RDKit Mol
        Scaffold molecule.
    similar_dfs : dict
        Dictionary of DataFrames, keyed by attachment point, each containing 'new_mol' and 'frag_att'.
    att_to_C_conversion : function
        Function that converts a fragment molecule by setting the correct attachment atom.

    Returns
    -------
    final_mol_df : pd.DataFrame
        DataFrame containing cleaned SMILES strings of all generated molecules.
    """
    final_mols = []

    for c in combinations:
        mol_num_i = scaffold_mol.GetNumAtoms()

        for i, mol_i in enumerate(c):
            att_pt = list(similar_dfs.keys())[i]
            frag_i = similar_dfs[att_pt].loc[mol_i, 'new_mol']
            frag_i = att_to_C_conversion(frag_i, None, [int(similar_dfs[att_pt].loc[mol_i, "frag_att"])])

            if i == 0:
                frag_att = mol_num_i + similar_dfs[att_pt].loc[mol_i, "frag_att"]
                combo = Chem.CombineMols(scaffold_mol, frag_i)
                combo_editable = Chem.EditableMol(combo)
                combo_editable.AddBond(int(att_pt), int(frag_att), order=Chem.rdchem.BondType.SINGLE)
            else:
                new_mol_num_i = combo_editable.GetMol().GetNumAtoms()
                frag_att = new_mol_num_i + similar_dfs[att_pt].loc[mol_i, "frag_att"]
                combo = Chem.CombineMols(combo_editable.GetMol(), frag_i)
                combo_editable = Chem.EditableMol(combo)
                combo_editable.AddBond(int(att_pt), int(frag_att), order=Chem.rdchem.BondType.SINGLE)

        combined_mol = combo_editable.GetMol()
        Chem.SanitizeMol(combined_mol)
        final_mols.append(combined_mol)

    # remove atom map numbers
    for mol in final_mols:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)

    smi = [Chem.MolToSmiles(mol) for mol in final_mols]

  
    def clean_bracketed_carbons(smiles: str) -> str:
        fixed = re.sub(r"\[\d*C\]", "C", smiles)  # replace [C], [4C], [5C], etc. with C
        mol = Chem.MolFromSmiles(fixed)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)

    cleaned_smi = [clean_bracketed_carbons(x) for x in smi]

    # make DataFrame
    final_mol_df = pd.DataFrame({"Smiles": cleaned_smi})

    return final_mol_df
