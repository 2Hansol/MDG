import sys
sys.path.append("/data/neural-head-avatars")


from optimization.train_pl_module import train_pl_module
from util.log import get_logger
from data.real import RealDataModule
from models.mdg_optimizer import MDGOptimizer

logger = get_logger("nha", root=True)

if __name__ == "__main__":
    train_pl_module(MDGOptimizer, RealDataModule)
