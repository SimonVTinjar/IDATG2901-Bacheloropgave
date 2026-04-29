from pytorch_fid import fid_score
import torch


def main():
    real_path = "full_clean_384x192"
    generated_path = "generated_images384x192"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    fid_value = fid_score.calculate_fid_given_paths(
        [real_path, generated_path],
        batch_size=32,
        device=device,
        dims=2048,
        num_workers=0,  # viktig på Windows
    )

    print(f"FID score: {fid_value:.2f}")


if __name__ == "__main__":
    main()