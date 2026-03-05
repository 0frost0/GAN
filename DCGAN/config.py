
class Config:
    DataRoot = '../data'
    output_dir = "dcgan_results"
    model_save_dir = "checkpoints"
    # Number of workers to use to load the dataset using loaders
    workers = 2
    #[train]
    random_seed = 100
    batch_size = 32
    image_size = 28
    num_channels = 1
    num_epochs = 80
    learning_rate = 0.0002
    # optimizer parameters
    beta1 = 0.5
    # gpu parameters
    num_gpu = 1
    #[generator]
    latent_dim = 100
    # Size of feature maps in generator
    nfg = 32
    #[discriminator]
    nfd = 32

