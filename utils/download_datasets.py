import os
import gdown

def download_file(file_id, output):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)

def download_folder(folder_id, output):
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(url, output=output, quiet=False)

def main():
    os.makedirs("datasets", exist_ok=True)

    print("Downloading latent_dataset.zip...")
    download_file("1QMnCbjiuMjxfr5VVt8CL05YBanG6vDoS", "datasets/latent_dataset.zip")

    print("Downloading celeba_dataset.zip...")
    download_file("1Cw5FQUUSs8KO36m4mPS-gSOT-HngP4QC", "datasets/celeba_dataset.zip")

    print("Downloading annotations folder...")
    download_folder("1syoG8p9UFlUhb9B3zbweCOUtMO0ubuJ5", "datasets/annotations")

    print("All downloads completed.")

if __name__ == "__main__":
    main()
