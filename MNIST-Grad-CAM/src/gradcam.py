import numpy as np 
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt  
from PIL import Image
from pytorch_grad_cam import GradCAM as TorchGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from models import ConvolutionalNetwork

dtype = torch.float32

torch.manual_seed(1)


class MNISTGradCAM(nn.Module):
   

    def __init__(self, network):

        super().__init__()
        self.net = network 
        self.classes = range(10)
        self.gradients = None 

        # breaking down the network into usable chunks. 
        self.CONV = self.net[0]

        self.FC = self.net[1] 




    def L_GC(self, image_data, class_name):
        image_batch, label_batch = image_data  # e.g. [64, 1, 28, 28]

        # Find matching target samples
        for i in range(len(label_batch)):
            if class_name == label_batch[i]:
                input_tensor = image_batch[i].unsqueeze(0)  # shape: [1, 1, 28, 28]
                break
        else:
            print("Target class not found.")
            return None

        # Select the target layer (the last Conv layer)
        # Note: self.CONV is nn.Sequential, and the target layer is the last layer
        target_layers = [self.CONV[-1]]  # Usually the last Conv/ReLU layer

        # Using the GradCAM interface
        cam = TorchGradCAM(model=self.net, target_layers=target_layers)

        # Specifying the classification target (class index)
        targets = [ClassifierOutputTarget(class_name)]

        # Get heatmap
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        # Visualization (input image normalized to [0,1])
        input_image = input_tensor.squeeze().cpu().numpy()  # shape: [28,28]
        input_image_rgb = np.stack([input_image] * 3, axis=-1)  # Grayscale image converted to RGB format
        input_image_rgb = (input_image_rgb - input_image_rgb.min()) / (input_image_rgb.max() + 1e-6)

        visualization = show_cam_on_image(input_image_rgb, grayscale_cam, use_rgb=True)

        return grayscale_cam, visualization
"""
    def plotGCAM(self, gcscores):
        # image = image_batch[0][0].detach().numpy()
        # image is the [28x28] tensor 
        L_c = gcscores
        sal = Image.fromarray(L_c)
        sal.resize(image.shape, resample=Image.LINEAR)

        plt.title("test")
        plt.axis("off")
        # plt.imshow(image)
        plt.imshow(np.array(sal), alpha=0.5, cmap="jet")
        plt.show()
        
"""        

        
        

    






if __name__ == "__main__":


    CNN = ConvolutionalNetwork()



    a = torch.randn(64, 1, 28, 28, dtype=dtype, requires_grad=True)
    b = torch.ones(1, 28, 28, dtype=dtype, requires_grad=True)

    a_labels = torch.ones(64)
    

    # testing for one image tensor of size (1,28,28)
    gradcamtest = MNISTGradCAM(network=CNN)
    # print(gradcamtest.L_GC(image_data=(a, a_labels), class_name=1))






  



    

    

    
    



    