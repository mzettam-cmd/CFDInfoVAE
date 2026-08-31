"""
Main program

NOTE: The "run" in running mode here means we do both train and infer 

@yuningw

modified by 

@mze
"""

import      torch
import      argparse
import random
import numpy as np
import torch
from        lib             import init, POD
from        lib.runners     import  latentRunner,infoVAERunner
from        utils.figs_time import  vis_temporal_Prediction_InfoVAE
from        utils.figs      import  vis_pod, vis_infovae


# ============================================================
# Reproducibility
# ============================================================
seed = 42

random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True)

print(f"Random seed: {seed}")

# ============================================================
# End Reproducibility
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument('-nn',default="easy", type=str,   help="Choose the model for time-series prediction: easy, self OR lstm")
parser.add_argument('-re',default=40,     type=int,   help="40 OR 100, Choose corresponding Reynolds number for the case")
parser.add_argument('-m', default="test", type=str,   help='Switch the mode between train, infer and run')
parser.add_argument('-t', default="pre",  type=str,    help='The type of saved model: pre/val/final')
parser.add_argument('-pod',default=True, type=bool,    help='Compute POD')
args  = parser.parse_args()

device = ('cuda' if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    import os
    ## Env INIT
    datafile = init.init_env(args.re)
    print("datafile =", datafile)
    datafile ='data/Data2PlatesGap1Re40_Alpha-00_downsampled_v6.hdf5'
    print("datafile =", datafile)

     ## Info-VAE
    infvae= infoVAERunner(device,datafile)
    if args.m == 'train':
        infvae.train()
    elif args.m == 'test':
        infvae.infer(args.t)
    elif args.m == 'run':
        infvae.run()
    
    """
    ## POD
    if args.pod:
        POD = POD.POD(datafile, n_test=infvae.config.n_test, re=args.re,
                    path='res/', n_modes=10, delta_t=infvae.config.delta_t)
        POD.load_data()
        POD.get_POD()
        POD.eval_POD()
    
    
    """
    # Time-series prediction runner 
    lruner = latentRunner(args.nn,device)
    if args.m == 'train':
        lruner.train()
    elif args.m == 'test':
        lruner.infer(args.t)
    elif args.m == 'run':
        lruner.train()
        lruner.infer(args.t)
       
    vis_infovae(init.pathsBib.res_path + "modes_" + infvae.filename + ".hdf5",
            init.pathsBib.log_path + infvae.filename)
    #vis_pod(POD)
    vis_temporal_Prediction_InfoVAE(model_type=args.nn, predictor=lruner, vae=infvae)     


