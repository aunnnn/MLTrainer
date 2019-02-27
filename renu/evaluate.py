import matplotlib.pyplot as plt
import pandas as pd

def plot_losses_from_files(output_dir,files,start,plot_title="LSTM Performance"):
    legend_list=[]
    for loss_type in files:    
        df = pd.read_csv(files[loss_type])
        y = range(len(df))
        loss = list(df['loss'])
        if loss:
            plt.plot(y[start:],loss[start:])
            legend_list.append(loss_type)
        
    plt.legend(legend_list)
    plt.xlabel("# of minibatches")
    plt.ylabel("loss value")
    plt.title(plot_title)
    plt.savefig("./output/{}/{}.png".format(output_dir,"lossestest_plot"), dpi=80)
    plt.show()
    
def test(model, criterion, inputs, targets, output_dir):
    model.eval()
    total_loss=0.0
    
    for i in range(len(inputs)):

        outputs = model(inputs[i])

        loss = criterion(outputs, targets[i].long())

        # Add this iteration's loss to the total_loss
        total_loss+=loss.item()
        
    total_loss/=len(inputs)

    return total_loss
