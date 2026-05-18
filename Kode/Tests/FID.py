from pytorch_fid import fid_score
import torch
import json

def main():
    real_path = "F:\\BachmapperMedBilder\\Datasett\\170_usd_391x366"
    generated_path = "F:\\BachmapperMedBilder\\Generert\\Textdiffuser\\Klippet"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fid_value = fid_score.calculate_fid_given_paths(
        [real_path, generated_path],
        batch_size=32,
        device=device,
        dims=2048,
        num_workers=0,
    )

    results = {"fid": round(fid_value, 4)}

    with open("fid_basedataset_384x160asdsad.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"FID score: {fid_value:.4f}")
    print("Lagret til fid_basedataset_384x160asda.json")

if __name__ == "__main__":
    main()