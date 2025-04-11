# **Implementing Grad-CAM on a MNIST Classification Model**

_This project aims at implementing **Grad-CAM** on a simple Convolutional Neural Network_

------------------------------------
## **1. About the Project**

* The MNIST databse [1] of handwritten digits has a training set of 60,000 examples. Each sample is a 28x28 black-and-white image, and one class per digit.  

While a classification model featuring a (relatively simple) Convolutional Neural Network will be built, the main purpose is to implement **Grad-CAM** for self-edification

### **Grad-CAM, or _Gradient-Weighted Class Activation Mapping_.**
Grad-CAM is a a class discriminative visualization technique [3], meaning that it produces a visualization which highlights the regions of the input image that are important for its classification into a certain label.  

For Example, given that the class of the image is "Dog", Grad-CAM is able to identify the parts of the image which contribute most to the identification of the image as that of a "Dog". 

------------------------------------------------
## **2. Modus Operandi**
_Elaborating on the  process of Model building and Grad-CAM_

Crudely:
1. Create a ConvNet to classify MNIST.
2. Measure how it performs.
3. Implement Grad-CAM.
4. Use Grad-CAM to visualize the important neurons for assigning a class.  
5. Report Results.

### **_ConvNet_**
    
While MNIST classification is the very first thing everyone cuts their teeth on, it is still useful to revisit the dataset with more sophistication, both as a programmer and as a practitioner. 

Structure of the ConvNet built: 
~~~   
  1. Conv2d(1, 12, kernel_size=(5, 5), stride=(1, 1), padding=(None, None))
  2. ReLU()
  3. MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  4. Flatten(start_dim=1, end_dim=-1)
  5. Linear(in_features=128, out_features=10, bias=True)
  6. Sigmoid()
  7. Linear(in_features=1, out_features=2, bias=True)
  8. Softmax(dim=None)
~~~

### **_2.2 Grad-CAM_**

Grad-CAM computes _neuron importance scores_ ($\alpha$) for each class $c$ and each filter $k$ [2]. $\alpha$ for class $c$, and filter $k$ is given by: 
$$\alpha^{c} = \frac{1}{Z}\Sigma_{i}\Sigma_{j} \frac{\partial y^{c}}{\partial A_{ij}^{k}}$$

With the Class Discriminative Localization map being given by: 
$$L^{c}= ReLU(\Sigma_{k} \alpha_{k}^{c}A_k)$$

A (qualitative) algorithm used to process Grad-CAM:

  1. Break network into two parts: CONV (Convolutional) and FC (Fully Connected)
  2. Get feature maps (A_c) by taking output of CONV
  3. Get Class scores (y_c) by taking output of FC. IN the actual implementation, this will be the class prediction of the model.
    
  4. Calculate gradients of the clas scores with respect to feature maps using torch.autograd.grad(y_c, A_c).
  5. Find neuron importance scores (alpha) by global average pooling the gradients in step 4. 
  6. Reshape the activation maps to facilitate dot product
  7. Calculate the dot product of alpha_c and A_c, and apply ReLU() to the product.
   
  Et Voila! 

New Problem: 
  1. The model takes a batch input of size [64, 1, 28, 28]
  2. REWRITE L_C in gradcam.py FOR A BATCH OF IMAGES, then index the desired scores with the desired class index











--------------------------------------


## **3. Results and Discussion**


-----------------------------------
## **References**
[1] : [The MNIST Dataset, from Yann LeCun's website](http://yann.lecun.com/exdb/mnist/)

[2]: Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. and Batra, D. (2020). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. _International Journal of Computer Vision_, [online] 128(2), pp.336–359. doi:10.1007/s11263-019-01228-7.

‌[3] : [The Grad-CAM Website](http://gradcam.cloudcv.org/)


----------------------------------------
## **Notes**

* __Hooks__: `PyTorch Hooks` were used to obtain the Activation maps from the desired layer. 
    "Some operations need intermediary results to be saved during the forward pass in order to execute the backward pass. You can define how these saved tensors should be packed / unpacked using hooks" 
1. Hooks are functions that automatically execute after a particular event. 
2. PyTorch Hooks are registered for each Tensor or nn.Module object and are triggered by either the forward or backward pass of the object. 
3. Each hook can modify the input, output or internal Module parameters. Most commonly, they are used for debugging. 
4. The hook will be called everytime a gradient with respect to the Tensor is computed. 
5. **We have to register the backward hook to the activation map of the last CONV layer in the model** In my case, it is going to be `CONV2`, including the activation `ReLU()` 
6. It is also important to register the hook inside the forward() method, to avoid the issue of registering a hook to a duplicate tesnor and susequently losing the gradient. 
7. Checkout: 

    `from torchvision.models import vgg19`

    `vgg = vgg19(pretrained=True)`

    `print(vgg.features[:36])`


8. The Hook is registered on the input value. 





* __Gradients:__ .backward() called on a torch.Tensor calculates gradients. 
  1. Call .backward() on the most probable logit, which is obatained by performing a single forward pass of the image through the network. 
  2. Pytorch only caches the gradients of the leaf nodes in the computational graph, such as weights, biases and other parameters 


* **class GradCAM:** givving the entire class a sample tensor as input (even though the init doesnt accept it as an argument) gives the same output as if the tensor was fed into the CNN, which _is_ an argument of the class. STRANGE.
