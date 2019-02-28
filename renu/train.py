import torch
from LSTM import LSTM
import Dataloader
import os
import pandas as pd
import sys

def train(model, criterion, optimizer, inputs, targets, val_inputs, val_targets, output_dir, num_epochs=5):
    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()

    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device("cuda")
        extras = {"num_workers": 1, "pin_memory": True}
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")
    model = model.to(computing_device)
    print("Model on CUDA?", next(model.parameters()).is_cuda)
    
    val_losses_file = "./output/{}/validation_loss.csv".format(output_dir)
    train_losses_file = "./output/{}/train_loss.csv".format(output_dir)
    
    output_path=os.path.join('output',output_dir)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    with open(train_losses_file,"w") as f:
        f.write("epoch,minibatch,loss\n")
    f.close()
    with open(val_losses_file,"w") as f:
        f.write("epoch,minibatch,loss\n")
    f.close()

    
    
    N = 50
    N_minibatch_loss = 0.0

    # Track the loss across training
    total_loss = []
    avg_minibatch_loss = []
    val_losses = []
    val_count=0
    save_state_dict=None

    for epoch in range(num_epochs): 
        model.clear_hidden() # zero out hidden/memory state

        for i in range(len(inputs)):
            model.train()
            
            # Put the minibatch data in CUDA Tensors and run on the GPU if supported
            src, trg = inputs[i].to(computing_device), targets[i].to(computing_device)

            # Zero out gradient
            optimizer.zero_grad()

            outputs = model(src,computing_device)

            loss = criterion(outputs, trg)

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
                
                with open(train_losses_file,"a") as f:
                    f.write("{},{},{}\n".format(epoch,i,N_minibatch_loss))
                f.close()

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
                
                with open(val_losses_file,"a") as f:
                    f.write("{},{},{}\n".format(epoch,i,val_loss))
                f.close()

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

    #scores_file = "./output/{}/scores.csv".format(model_name)

    #train_loss_df.to_csv(train_loss_file)
    #val_loss_df.to_csv(val_loss_file)
    
    return train_loss_df, val_loss_df

if __name__ == '__main__':
	if len(sys.argv) != 4:
		print("need 3 arguments: epochs output_dir hidden_dim")
		sys.exit()

	# run train
	model = LSTM(hidden_dim=int(sys.argv[3]))
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters())#, lr=learning_rate)
	EPOCHS = int(sys.argv[1])
	output_dir=sys.argv[2]


	dataloader = Dataloader.Dataloader()
	train_inputs,train_targets = dataloader.load_data('../pa4Data/test.txt')
	val_inputs,val_targets = dataloader.load_data('../pa4Data/val.txt')


	train(model, criterion, optimizer, train_inputs, train_targets, val_inputs, val_targets, output_dir, num_epochs=EPOCHS)
