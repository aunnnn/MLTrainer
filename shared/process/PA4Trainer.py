import time
import torch
import os
import json
import numpy as np
import pathlib
import pprint
    

class PA4Trainer:
    """
    Main class for the whole train/validate/save process. 
    
    To use:    
        trainer = PA4Trainer(...)
        trainer.start()
        
    This will save following files automatically (under the session name folder):
    ----------------------------------------------------------------------------
    - config.json
        - the config dict provided to the trainer.
        
    - test_loss.npy
    
    - epoch_losses.npy
        - all epoch losses
    
    - v_interval_val_losses.npy
        - val loss every `config['validate_every_v_epochs']` epochs
        
    - v_interval_train_losses.npy 
        - train loss every `config['validate_every_v_epochs']` epochs 
        - (so we can easily plot it together with val loss)
    
    - model_state.pt
        - torch model state after complete training
        
    - model_state_min_val_so_far.pt
        - if use early stopping, this will be saved everytime validation loss is less than min
    
    - misc.json
        - dump anything else to save here after training, e.g., `is_early_stopped`
            
    Checkout __save_result() for more detail.
        
    """
    
    def __init__(self, model, criterion, optimizer, all_loaders, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.all_loaders = all_loaders
        self.config = config
        
        self.start_time = None        
        self.all_epoch_losses = [] # for all epochs
        
        self.v_interval_train_losses = [] # just for every v, to easily compare plotting with v
        self.v_interval_val_losses = [] # just for every v
        self.test_loss = None
                
        self.computing_device = get_computing_device()
        self.model.to(self.computing_device)
                
        # CONFIG
        self.SESSION_NAME = config['session_name']
        self.PATH_TO_SAVE_RESULT = config['path_to_save_result']
        
        self.N_EPOCHS = config['n_epochs']
        self.PRINT_EVERY_N_EPOCHS = config['print_every_n_epochs']
        self.VALIDATE_EVERY_V_EPOCHS = config['validate_every_v_epochs']
        self.VERBOSE = config['verbose']
        self.NUM_EPOCHS_NO_IMPROVEMENT_EARLY_STOP = config['num_epochs_no_improvement_early_stop']
        self.USE_EARLY_STOP = config['use_early_stop']
        self.PASS_HIDDEN_STATE_BETWEEN_EPOCHS = config['pass_hidden_states_between_epochs']
        
        self.save_folder_path = os.path.join(self.PATH_TO_SAVE_RESULT, self.SESSION_NAME)
        pathlib.Path(self.save_folder_path).mkdir(parents=True, exist_ok=True)
        
        self.EARLY_STOP_SAVE_PATH = os.path.join(self.save_folder_path, 'model_state_min_val_so_far.pt')
        
        # Save config
        with open(os.path.join(self.save_folder_path, 'config.json'), 'w') as fp:
            def string_var_config(var):
                return "{0}".format(var)
                
            saved_config = {
                'model': string_var_config(model),
                'criterion': string_var_config(criterion),
                'optimizer': string_var_config(optimizer),
                **config
            }
            json.dump(saved_config, fp, indent=4)
            
            if self.VERBOSE:
                print("Coverted model to device: {0}".format(self.computing_device))
                print('-----------')
                print("Trainer Config:")
                pprint.PrettyPrinter(indent=4).pprint(saved_config)
                print('-----------')
        
    def start(self):
        """
        Start the whole complete process.
        """
        self.__train()
        self.test_loss = self.__get_test_loss()
        self.__save_result()
        
        if self.VERBOSE:
            print("Done successfully!")
            print('---------------------')
    
    def __train(self):
        """
        Training phase
        """
        print("Start training...")
        
        train_loader = self.all_loaders['train']
        
        self.test_loss = None
        self.start_time = time.time()
        self.all_epoch_losses = []
        
        self.v_interval_train_losses = []
        self.v_interval_val_losses = []
        
        # Early stopping
        min_val_loss = float('inf')
        prev_val_loss = float('inf')
        consecutive_no_improvement_epochs = 0
        
        self.is_early_stopped = False
        
        # Set initial hidden
        self.model.reset_hidden(self.computing_device)

        for i_epoch in range(self.N_EPOCHS):
            
            if not self.PASS_HIDDEN_STATE_BETWEEN_EPOCHS:
                # Reset hidden to zeros for new epoch
                self.model.reset_hidden(self.computing_device)
            
            current_epoch_loss = 0.0

            for (inputs, labels) in train_loader:
                inputs, labels = inputs.to(self.computing_device), labels.to(self.computing_device)
                loss = self.__train_one_chunk(inputs, labels)
                current_epoch_loss += loss

            # Loss avg by chunks
            epoch_avg_train_loss = current_epoch_loss/len(train_loader)
            
            # Averaged by number of chunks, last one will be a bit skewed.
            self.all_epoch_losses.append(epoch_avg_train_loss)
            
            if i_epoch % self.VALIDATE_EVERY_V_EPOCHS == 0:
                self.v_interval_train_losses.append(epoch_avg_train_loss)
                val_loss = self.__get_validation_loss()
                self.v_interval_val_losses.append(val_loss)
                
                print('Epoch {0}, validation loss: {1}'.format(i_epoch, val_loss))
                
                # EARLY STOPPING
                if self.USE_EARLY_STOP:      
                    if val_loss < min_val_loss:
                        consecutive_no_improvement_epochs = 0
                        min_val_loss = val_loss
                        # Save model
                        torch.save(self.model.state_dict(), self.EARLY_STOP_SAVE_PATH)
                    else:
                        consecutive_no_improvement_epochs += 1
                        if self.VERBOSE:
                            print("Validation loss increased for {0} epochs...".format(consecutive_no_improvement_epochs))
                        if consecutive_no_improvement_epochs == self.NUM_EPOCHS_NO_IMPROVEMENT_EARLY_STOP:
                            ###################
                            ## EARLY STOPPED! #
                            ###################
                            # Restore model before exit
                            self.model.load_state_dict(torch.load(self.EARLY_STOP_SAVE_PATH))
                            self.model.eval()
                            print('Stop training as validation loss increases for {} epochs.'.format(self.NUM_EPOCHS_NO_IMPROVEMENT_EARLY_STOP))
                            self.is_early_stopped = True
                            break

            if i_epoch % self.PRINT_EVERY_N_EPOCHS == 0:
                print('Epoch %d, %d%% (%s) train loss: %f' % (i_epoch, i_epoch / self.N_EPOCHS * 100, time_since(self.start_time), epoch_avg_train_loss))
            
    def __get_test_loss(self):        
        test_loader = self.all_loaders['test']
        return self.__get_full_loss(test_loader)
        
    def __get_validation_loss(self):
        val_loader = self.all_loaders['val']
        return self.__get_full_loss(val_loader)                       
                
    def __get_full_loss(self, loader):
        """
        Evaluate loss on the whole dataset from loader. (with torch.no_grad)
        """
        total_val_loss = 0
        with torch.no_grad():
            for (inputs, labels) in loader:
                inputs, labels = inputs.to(self.computing_device), labels.to(self.computing_device)

                loss = self.__evaluate_loss_one_chunk(inputs, labels)
                total_val_loss += loss
                
        # Loss avg by chunks, a bit skewed last chunk
        return total_val_loss/len(loader)
        
                
    def __train_one_chunk(self, inputs, labels):
        
        # If [chunk_size, num_features], make it [1, chunk_size, num_features]
        if len(inputs.size()) == 2:
            inputs.unsqueeze_(dim=0)

        if len(labels.size()) == 2:
            labels.unsqueeze_(dim=0)

        # Truncated BPTT
        self.model.detach_hidden()    
        self.optimizer.zero_grad()

        # Only one batch
        logits = self.model(inputs)
        labels = labels[0]    

        # Turn into non-zero indices
        label_inds = labels.topk(1, dim=1)[1].view(-1)
        cross_entropy_loss = self.criterion(logits, label_inds)

        cross_entropy_loss.backward()    
        self.optimizer.step()

        return cross_entropy_loss.item()
    
    def __evaluate_loss_one_chunk(self, inputs, labels):
        
        # If [chunk_size, num_features], make it [1, chunk_size, num_features]
        if len(inputs.size()) == 2:
            inputs.unsqueeze_(dim=0)

        if len(labels.size()) == 2:
            labels.unsqueeze_(dim=0)

        # Only one batch
        logits = self.model(inputs)
        labels = labels[0]    

        # Turn into non-zero indices
        label_inds = labels.topk(1, dim=1)[1].view(-1)
        cross_entropy_loss = self.criterion(logits, label_inds)
        return cross_entropy_loss.item()        
        
        
    def __save_result(self):
        """
        Save all results to file.
        """
        
        if self.VERBOSE:
            print('-----------')
            print("Saving result...")
            
        folder_name = self.save_folder_path
        
        # Model
        model_path = os.path.join(folder_name, 'model_state.pt')
        torch.save(self.model.state_dict(), model_path)
        
        if self.VERBOSE:
            print("-> model saved.")
        
        # Report
        test_loss_path = os.path.join(folder_name, 'test_loss.npy') # Just a single number                                   
        all_epoch_losses_path = os.path.join(folder_name, 'epoch_losses.npy')
        v_interval_train_losses_path = os.path.join(folder_name, 'v_interval_train_losses.npy')
        v_interval_val_losses_path = os.path.join(folder_name, 'v_interval_val_losses.npy')
        
        np.save(test_loss_path, np.array([self.test_loss]))
        np.save(all_epoch_losses_path, np.array(self.all_epoch_losses))
        np.save(v_interval_train_losses_path, np.array(self.v_interval_train_losses))
        np.save(v_interval_val_losses_path, np.array(self.v_interval_val_losses))
                                  
        if self.VERBOSE:
            print("-> report saved.")
                                  
                                  
        misc_path = os.path.join(folder_name, 'misc.json')
        misc_content = {
            'is_early_stopped': self.is_early_stopped
        }
                                  
        with open(misc_path, 'w') as fp:
            json.dump(misc_content, fp, indent=4)
        if self.VERBOSE:
            print("-> misc.json saved.")                
    
# Utilities
def time_since(since):
    now = time.time()
    s = now - since
    m = np.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def get_computing_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")