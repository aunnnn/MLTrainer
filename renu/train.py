import torch
from LSTM import LSTM
import Dataloader
import os
import pandas as pd

def train(model, criterion, optimizer, inputs, targets, val_inputs, val_targets, output_dir, num_epochs=5):
    output_path=os.path.join('output',output_dir)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    N = 50
    N_minibatch_loss = 0.0

    # Track the loss across training
    total_loss = []
    avg_minibatch_loss = []
    val_losses = []
    epochs = []
    minibatches = []
    val_count=0
    save_state_dict=None

    for epoch in range(num_epochs): 
        model.clear_hidden() # zero out hidden/memory state

        for i in range(len(inputs)):
            model.train()

            # Zero out gradient
            optimizer.zero_grad()

            outputs = model(inputs[i])

            loss = criterion(outputs, targets[i].long())

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Add this iteration's loss to the total_loss
            total_loss.append(loss.item())
            N_minibatch_loss += loss.item()

            if i % N == 0: 
                # Print the loss averaged over the last N mini-batches    
                N_minibatch_loss /= N
                print('Epoch %d, average minibatch %d loss: %.3f' %
                    (epoch + 1, i, N_minibatch_loss))

                # Add the averaged loss over N minibatches and reset the counter
                avg_minibatch_loss.append(N_minibatch_loss)
                N_minibatch_loss = 0.0

                # validation
                print('...validating...')
                val_loss=0.0
                with torch.no_grad():
                    for v in range(len(val_inputs)):
                        val_outputs = model(val_inputs[v])
                        val_loss += criterion(val_outputs, val_targets[v]).item()
                val_loss/=len(val_inputs)
                val_losses.append(val_loss)

                epochs.append(epoch+1)
                minibatches.append(i)

                print('Epoch %d, average validation loss: %.3f' %
                    (epoch + 1, val_loss))

                # early stopping
                if len(val_losses)>=2 and val_loss>val_losses[-2]:
                    if val_count==0:
                        save_state_dict = model.state_dict()
                    val_count+=1
                    if val_count==5:
                        break  

        print("Finished", epoch + 1, "epochs of training")

    ### save model ###
    if save_state_dict:
        PATH = "./output/{}/best.pt".format(output_dir)
        torch.save(save_state_dict, PATH)
    PATH = "./output/{}/final.pt".format(output_dir)
    torch.save(model.state_dict(), PATH)
    
    ### save train and val losses ###
    train_loss_df = pd.DataFrame({'loss':avg_minibatch_loss})
    val_loss_df = pd.DataFrame({'loss':val_losses})

    val_loss_file = "./output/{}/validation_loss.csv".format(output_dir)
    train_loss_file = "./output/{}/train_loss.csv".format(output_dir)
    #scores_file = "./output/{}/scores.csv".format(model_name)

    train_loss_df.to_csv(train_loss_file)
    val_loss_df.to_csv(val_loss_file)
    
    