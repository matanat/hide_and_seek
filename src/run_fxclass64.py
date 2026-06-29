from templates import *

if __name__ == '__main__':
    # Train the DAE.
    # Set conf.data_manifest_path to your Excel manifest before running.
    # Required columns: image, seg_msk, label (0=healthy, 1=malignant)
    gpus = [0]
    conf = fxclass64_autoenc()
    conf.data_manifest_path = '/path/to/your/dataset.xlsx'
    train(conf, nodes=1, gpus=gpus)
