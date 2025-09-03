import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem import BRICS

def main(input_csv, output_csv):
    
    df = pd.read_csv(input_csv)

    new_rows = []

    for smi in df["Smiles"]:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
           
            frags = BRICS.BRICSDecompose(mol)
            for frag in frags:
                new_rows.append({"Parent_Smile": smi, "Fragment": frag})

        except Exception:
            continue

    new_df = pd.DataFrame(new_rows)

    new_df.to_csv(output_csv, index=False)
    print(f"Saved file with fragments to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BRICS decomposition of molecules from a CSV file.")
    parser.add_argument("input_csv", help="Path to input CSV file with a 'Smiles' column")
    parser.add_argument("output_csv", help="Path to save output CSV with BRICS fragments")
    
    args = parser.parse_args()
    main(args.input_csv, args.output_csv)

