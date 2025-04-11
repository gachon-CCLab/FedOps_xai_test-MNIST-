import numpy as np 
import os 
import torch
torch.manual_seed(0)
import torch.nn as nn 
import matplotlib.pyplot as plt 


# Custom Imports
from src.loadvis import Loader
from src.loadvis import Visualizer
from src.models import ConvolutionalNetwork
from src.traintest import TrainTest
from src.gradcam import MNISTGradCAM

if __name__ == "__main__":
    loader = Loader(PATH="/home/ccl/Desktop/ECG_Anaylzer/MNIST/dataset/mnist/MNIST/raw/", batch_size=64)

    #1. Loading the data 
    trainloader, testloader = loader.getdata()




    # print(getanimage(loader=trainloader, class_idx=0, plot=False))



    
    # #2. Visualizing the data
    visualizer = Visualizer()
    # visualizer.visualize(trainloader)


    # #3. Building the network 
    model = ConvolutionalNetwork()
    cnn = model.network()


    # #4. Initial Training cycle and Getting baselines
    traintest = TrainTest(network=cnn, num_epochs=1, learning_rate=0.01)

    # traintest.train_one_epoch(trainloader, epoch=0)
    trained_network, training_loss = traintest.train_all_epochs(trainloader)


    # 5. Testing the model on new data
    traintest.test(network = model, loader = testloader)

    # 6. Plotting the loss curve for the training cycle
    # visualizer.plotperf(training_loss, num_epochs = 10)

    images, labels = next(iter(testloader))
    # 7. Obtaining Grad-CAM for the model 
    gradcam = MNISTGradCAM(network=cnn)  # 假设你封装在某个类里
    L_c, vis = gradcam.L_GC(image_data=(images, labels), class_name=1)

    import matplotlib.pyplot as plt
    plt.imshow(vis)
    plt.axis("off")
    plt.title("Grad-CAM Visualization")
    plt.show()

    # 保存图像到本地路径
    output_path = "./gradcam_output"
    os.makedirs(output_path, exist_ok=True)
    save_name = os.path.join(output_path, f"gradcam_class_{1}.png")

    plt.imsave(save_name, vis)
    print(f"✅ Grad-CAM image saved to {save_name}")





    




    




