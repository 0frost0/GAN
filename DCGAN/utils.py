import logging
import os

def get_logger(log_dir):
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    logger = logging.getLogger('DCGAN')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger