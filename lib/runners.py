"""
Runners for the VAE and temporal-dynamic prediction in latent space 
@yuningw
"""

import os 
import time
from pathlib import Path
import h5py
import numpy as np
import torch 
from torch          import nn

from lib.init       import pathsBib
from lib.train      import * 
from lib.model      import * 
from lib.pp_time    import * 
from lib.pp_space   import spatial_Mode
from lib.datas      import * 

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


"""
Runners for InfoVAE (minimal modification from beta-VAE)
"""



####################################################
### RUNNER for InfoVAE
####################################################

class infoVAERunner(nn.Module):
    def __init__(self, device, datafile) -> None:
        
        from configs.infovae import InfoVAE_config as cfg 
        from configs.nomenclature import Name_InfoVAE
        
        super(infoVAERunner,self).__init__()
        print("#"*30)

        self.config     = cfg
        self.filename   = Name_InfoVAE(self.config)
        
        self.datafile   = datafile
        self.device     = device

        #  CHANGEMENT ICI
        self.model      = get_infovae(self.config.latent_dim)
        
        self.model.to(device)

        self.fmat       =  '.pth.tar'

        print(f"INIT InfoVAE, device: {device}")
        print(f"Case Name:\n {self.filename}")

#-------------------------------------------------
    def run(self):
        self.train()
        self.infer(model_type='final')

#-------------------------------------------------
    def train(self):
        print("#"*30)
        print("INFO: Start Training ")
        self.get_data()
        self.compile()
        self.fit()
        self.train_dl   = None
        self.val_dl     = None
        print(f"INFO: Training finished, cleaned the data loader")
        print("#"*30)

#-------------------------------------------------
    def infer(self, model_type):
        print("#"*30)
        self.load_pretrain_model(model_type=model_type)
        print("INFO: Model has been loaded!")
        
        self.get_test_data()
        print("INFO: test data has been loaded!")
        self.post_process()

        print(f"INFO: Inference ended!")
        print("#"*30)


#-------------------------------------------------
    def get_data(self): 
        
        u_scaled, self.mean, self.std = loadData(self.datafile)
        
        u_scaled            = u_scaled[::self.config.downsample]
        n_total             = u_scaled.shape[0]
        self.n_train        = n_total - self.config.n_test
        
        print(f"INFO: Data Summary: N train: {self.n_train:d}," + \
                f"N test: {self.config.n_test:d},"+
                f"N total {n_total:d}")
        
        self.train_dl, self.val_dl = get_vae_DataLoader(
                                                            d_train=u_scaled,
                                                            n_train=self.n_train,
                                                            device= self.device,
                                                            batch_size= self.config.batch_size)

        print(f"INFO: Dataloader generated, Num train batch = {len(self.train_dl)} \n" +
              f"Num val batch = {len(self.val_dl)}")

#-------------------------------------------------
    def compile(self):

        from torch.optim import lr_scheduler
        
        print("#"*30)
        print(f"INFO: Start Compiling")

        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters())

        self.opt = torch.optim.Adam(
            [   {'params': encoder_params, 'weight_decay': self.config.encWdecay},
                {'params': decoder_params, 'weight_decay': self.config.decWdecay}], 
                lr=self.config.lr, weight_decay=0)
        
        self.opt_sch = lr_scheduler.OneCycleLR(
                                            self.opt, 
                                            max_lr=self.config.lr,
                                            epochs=self.config.epochs,
                                            steps_per_epoch=len(self.train_dl),
                                            div_factor=2, 
                                            final_div_factor=self.config.lr/self.config.lr_end, 
                                            pct_start=0.2)

        # NO MORE betaScheduler → remplacé par alpha/lambda
        self.alpha = self.config.alpha
        self.lam   = self.config.lam

        print(f"INFO: Compiling Finished!")

#-------------------------------------------------
    def fit(self):

        from torch.utils.tensorboard import SummaryWriter
        from utils.io import save_checkpoint

        print(f"Training {self.filename}")
        logger = SummaryWriter(log_dir=pathsBib.log_path + self.filename)

        bestloss = 1e6
        
        import os
        import torch

        start_epoch = 1

        checkpoint_file = f"{pathsBib.chekp_path}/{self.filename}_checkpoint.pth"

        if os.path.exists(checkpoint_file):
            print(f"Loading checkpoint: {checkpoint_file}")

            checkpoint = torch.load(checkpoint_file, map_location=self.device)

            self.model.load_state_dict(checkpoint['state_dict'])
            self.opt.load_state_dict(checkpoint['optimizer_dict'])
            self.opt_sch.load_state_dict(checkpoint['scheduler_dict'])

            start_epoch = checkpoint['epoch'] + 1
            bestloss = checkpoint['bestloss']

            print(f"Restarting from epoch {start_epoch}")
            
        for epoch in range(start_epoch, self.config.epochs + 1):

            self.model.train()

            # TRAIN INFOVAE
            loss, MSE, KLD, MMD, elapsed, collapsed = train_epoch_infovae(
                                                        model=self.model,
                                                        data=self.train_dl,
                                                        optimizer=self.opt,
                                                        alpha=self.alpha,
                                                        lam=self.lam,
                                                        device=self.device)

            self.model.eval()

            loss_test, MSE_test, KLD_test, MMD_test, elapsed_test = test_epoch_infovae(
                                                        model=self.model,
                                                        data=self.val_dl,
                                                        alpha=self.alpha,
                                                        lam=self.lam,
                                                        device=self.device)

            self.opt_sch.step()

            printProgress(epoch=epoch,
                            epochs=self.config.epochs,
                            loss=loss,
                            loss_test=loss_test,
                            MSE=MSE,
                            KLD=KLD,
                            elapsed=elapsed,
                            elapsed_test=elapsed_test,
                            collapsed=collapsed)

            #   LOGS
            logger.add_scalar('Loss/Total', loss, epoch)
            logger.add_scalar('Loss/MSE', MSE, epoch)
            logger.add_scalar('Loss/KLD', KLD, epoch)
            logger.add_scalar('Loss/MMD', MMD, epoch)

            logger.add_scalar('Loss_test/Total', loss_test, epoch)
            logger.add_scalar('Loss_test/MSE', MSE_test, epoch)
            logger.add_scalar('Loss_test/KLD', KLD_test, epoch)
            logger.add_scalar('Loss_test/MMD', MMD_test, epoch)

            logger.add_scalar('LR', self.opt_sch.get_last_lr()[0], epoch)
            
            
            # Save checkpoint every 20 epochs
            if epoch % 20 == 0:

                state = {
                    'epoch': epoch,
                    'bestloss': bestloss,
                    'state_dict': self.model.state_dict(),
                    'optimizer_dict': self.opt.state_dict(),
                    'scheduler_dict': self.opt_sch.state_dict()
                }

                ckp_file = f"{pathsBib.chekp_path}/{self.filename}_checkpoint.pth"

                save_checkpoint(state=state, path_name=ckp_file)

                print(f"Checkpoint saved at epoch {epoch}")

            if (loss_test < bestloss and epoch > 100):
                bestloss = loss_test

                checkpoint = {
                    'epoch': epoch,
                    'bestloss': bestloss,
                    'state_dict': self.model.state_dict(),
                    'optimizer_dict': self.opt.state_dict(),
                    'scheduler_dict': self.opt_sch.state_dict()
                }

                ckp_file = f'{pathsBib.chekp_path}/{self.filename}_bestVal' + self.fmat
                save_checkpoint(state=checkpoint, path_name=ckp_file)

                print(f'## Checkpoint saved at epoch {epoch}')
                
                
        if start_epoch > self.config.epochs:
          epoch = checkpoint['epoch']
          
        checkpoint = {
            'epoch': epoch,
            'bestloss': bestloss,
            'state_dict': self.model.state_dict(),
            'optimizer_dict': self.opt.state_dict(),
            'scheduler_dict': self.opt_sch.state_dict()
        }
        ckp_file = f'{pathsBib.chekp_path}/{self.filename}_final' + self.fmat
        save_checkpoint(state=checkpoint, path_name=ckp_file)

        print(f'Final model saved!')

#-------------------------------------------------
    def load_pretrain_model(self,model_type='pre'):

        model_type_all = ['pre','val','final']
        assert(model_type in model_type_all)

        if model_type == 'pre':
            model_path = os.path.join(pathsBib.pretrain_path,self.filename + self.fmat)
        elif model_type == 'val':
            model_path = os.path.join(pathsBib.chekp_path,self.filename + '_bestVal' + self.fmat)
        else:
            model_path = os.path.join(pathsBib.chekp_path,self.filename + '_final' + self.fmat)
            
        try:
            print("DEBUG pathsBib.chekp_path =", pathsBib.chekp_path)
            print("DEBUG self.filename =", self.filename)
            print("DEBUG self.fmat =", self.fmat)
            print("DEBUG model_path =", model_path)
            print("DEBUG exists =", os.path.exists(model_path))
            ckpoint = torch.load(model_path, map_location=self.device)
        except Exception as e:
            print(f"ERROR: Cannot load model: {model_path}")
            print(f"ERROR DETAILS: {e}")
            exit()


        self.model.load_state_dict(ckpoint['state_dict'])
        print(f'INFO: model loaded!')

#-------------------------------------------------
    def get_test_data(self):

        from torch.utils.data import DataLoader
        
        u_scaled, self.mean, self.std = loadData(self.datafile)
        
        u_scaled = u_scaled[::self.config.downsample]
        n_total  = u_scaled.shape[0]
        self.n_train = n_total - self.config.n_test
        
        self.train_d, self.test_d = u_scaled[:self.n_train], u_scaled[self.n_train:]

        self.train_dl = DataLoader(torch.from_numpy(self.train_d), batch_size=1)
        self.test_dl  = DataLoader(torch.from_numpy(self.test_d), batch_size=1)

        print(f"INFO: Test loader ready")

#-------------------------------------------------
    def post_process(self):

        fname = pathsBib.res_path + "modes_" + self.filename
        
        spatial_Mode(fname,
                    model=self.model,
                    latent_dim=self.config.latent_dim,
                    train_data=self.train_d,
                    test_data=self.test_d,
                    dataset_train=self.train_dl,
                    dataset_test=self.test_dl,
                    mean=self.mean,
                    std=self.std,
                    device=self.device)

    

####################################################
### RUNNER for Temporal-dynamics Prediction
####################################################


class latentRunner(nn.Module): 
    def __init__(self,name,device):
        """
        A runner for latent space temporal-dynmaics prediction

        Args:

            name            :       (str) The model choosed for temporal-dynamics prediction 

            device          :       (Str) The device going to use
            
        """

        super(latentRunner,self).__init__()
        print("#"*30)
        print(f"INIT temporal predictor: {name}, device: {device}")
        self.device = device
        self.model,self.filename, self.config = get_predictors(name)
        
        self.NumPara = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.fmat   = '.pt'
        print(f"INFO: The model has been generated, num of parameter is {self.NumPara}")
        print(f"Case Name:\n {self.filename}")


#-------------------------------------------------

    def train(self):
        print("#"*30)
        print("INFO: Start Training ")
        self.get_data()
        self.compile()
        self.fit()
        self.train_dl   = None
        self.val_dl     = None
        print(f"INFO: Training finished, cleaned the data loader")
        print("#"*30)

#-------------------------------------------------
    def infer(self, model_type = 'pre',
            if_window=True, 
            if_pmap=True):
        """
        
        Inference and evaluation of the model 

        Args: 

            model_type: (str) The type of model to load 

            if_window : (str) If compute the sliding-widnow error 

            if_pmap : (str) If compute the Poincare Map 
        
        """
        
        print("#"*30)
        print("INFO: Start post-processing")
        # self.com
        self.load_pretrain_model(model_type=model_type)

        self.post_process(if_window,if_pmap)
        print(f"INFO: Inference ended!")
        print("#"*30)

#-------------------------------------------------


    def get_data(self):
        """
        Get the latent space variable data for training and validation
        """ 
        try: 
            hdf5 = h5py.File(pathsBib.data_path + "latent_data.h5py")
            data   = np.array(hdf5['vector'])
        except:
            print(f"Error: DataBase not found, please check path or keys")

        X,Y = make_Sequence(self.config,data=data)
        self.train_dl, self.val_dl =make_DataLoader(torch.from_numpy(X),torch.from_numpy(Y),
                                                    batch_size=self.config.Batch_size,
                                                    drop_last=False, 
                                                    train_split=self.config.train_split)
        print(f"INFO: DataLoader Generated!")
        del data, X, Y

#-------------------------------------------------

    def compile(self): 
        """
        Compile the model with optimizer, scheduler and loss function
        """
        self.loss_fn =   torch.nn.MSELoss()
        self.opt     =   torch.optim.Adam(self.model.parameters(),lr = self.config.lr, eps=1e-7)
        self.opt_sch =  [  
                        torch.optim.lr_scheduler.ExponentialLR(optimizer=self.opt, gamma= (1 - 0.01)) 
                        ]

#-------------------------------------------------

    def fit(self): 
        """
        Training Model, we use the fit() function 
        """

        s_t = time.time()
        history = fitting(  self.device, 
                            self.model,
                            self.train_dl, 
                            self.loss_fn,
                            self.config.Epoch,
                            self.opt,
                            self.val_dl, 
                            scheduler=self.opt_sch,
                            if_early_stop=self.config.early_stop,
                            patience=self.config.patience)
        e_t = time.time()
        cost_time = e_t - s_t
        
        print(f"INFO: Training FINISH, Cost Time: {cost_time:.2f}s")
        
        check_point = { "model":self.model.state_dict(),
                        "history":history,
                        "time":cost_time}
        
        torch.save(check_point,pathsBib.model_path + self.filename + self.fmat)
        print(f"INFO: The checkpoints has been saved!")

#-------------------------------------------------


    def load_pretrain_model(self,model_type='pre'):
        """

        Load the pretrained model for infoVAE

        Args: 

            model_type  : ['pre', 'val','final']  (str) Choose from pre-trained, best valuation and final model 
        
        """
        
        model_type_all = ['pre','val','final']
        assert(model_type in model_type_all), print('ERROR: No type of the model matched')

        if model_type == 'pre':
            model_path = os.path.join(pathsBib.model_path, self.filename + self.fmat)
        elif model_type == 'val':
            model_path = os.path.join(pathsBib.chekp_path,self.filename + '_bestVal' + self.fmat)
        else:
            model_path = os.path.join(pathsBib.chekp_path,self.filename + '_final' + self.fmat)
        try:
            print("DEBUG model_path =", model_path)
            print("DEBUG exists =", os.path.exists(model_path))

            ckpoint = torch.load(model_path, map_location=self.device)

        except Exception as e:
            print("ERROR: Model NOT found!")
            print("ERROR model_path =", model_path)
            print("ERROR details =", e)
            exit()
        stat_dict   = ckpoint['model']

        self.model.load_state_dict(stat_dict)
        self.history = ckpoint['history']

        
        print(f'INFO: the state dict has been loaded!')
        print(self.model.eval)


#-------------------------------------------------

    def post_process(self,if_window=True,if_pmap=True):
        """
        Post Processing of the temporal-dynamics predcition 
        Args:
            
            if_window   :   (bool) If compute the sliding-window error 

            if_pmap     :   (bool) If compute the Poincare Map 
        """ 
        
        try: 
            hdf5        = h5py.File(pathsBib.data_path + "latent_data.h5py")
            test_data   = np.array(hdf5['vector_test'])
        except:
            print(f"Error: DataBase not found, please check path or keys or try run the vae first")

        print(f"INFO: Test data loaded, SIZE = {test_data.shape}")
        Preds = make_Prediction(test_data   = test_data, 
                                model       = self.model,
                                device      = self.device,
                                in_dim      = self.config.in_dim,
                                next_step   = self.config.next_step)
        
        if if_window: 
            print(f"Begin to compute the sliding window error")
            window_error = Sliding_Window_Error(test_data, 
                                                self.model, 
                                                self.device, 
                                                self.config.in_dim)
        else: 
            window_error = np.nan
        
        
        if if_pmap:
            planeNo      = 0 
            postive_dir  = True
            lim_val      = 2.5 # Limitation of x and y bound when compute joint pdf 
            grid_val     = 50
            InterSec_pred = Intersection(Preds,     planeNo=planeNo,postive_dir=postive_dir)
            InterSec_test = Intersection(test_data, planeNo=planeNo,postive_dir=postive_dir)
        else:
            InterSec_pred = np.nan
            InterSec_test = np.nan
        
        
        np.savez_compressed(
                            pathsBib.res_path + self.filename + ".npz",
                            p = Preds, 
                            g = test_data,
                            e = window_error,
                            pmap_g = InterSec_test,
                            pmap_p = InterSec_pred
                            )