# Image Noise Model

This repository demonstrates how to model the noise in a Digital Camera. It is based off of Digital Camera Noise Models from [1] and [2]. The noise in an image is not additive white Gaussian Noise as modeled in many situations, it is dependent upon the image intensity and it is spatially correlated due to the Bayer Color Filter Array and Demosaicing.

<br>

![image](https://user-images.githubusercontent.com/60835780/224213193-c5b7980f-0a4d-45b3-a74e-e05ceaafc6a1.png)

<br>
Compared to the iid noise, the camera model noise is more smooth and subtile and is harder to remove than the grainy iid noise.

<br>

## Autocorrelation 

![image](https://user-images.githubusercontent.com/60835780/224213369-250060a0-fa56-4889-adb4-b72a54161a20.png)

 ## Usage
 ```
     from image_noise_model import get_camera_noise

     camera_noise = get_camera_noise(image, s_sigma=0.01, c_sigma=0.02, c=3)
     noisy_image = image + camera_noise
     
 ```
 
<!-- 
<center>
  <img width="500"  align="middle" src="https://user-images.githubusercontent.com/60835780/224213369-250060a0-fa56-4889-adb4-b72a54161a20.png"/>
</center>
 -->
<br>
<br>
<br>

## References

[1] https://people.csail.mit.edu/billf/publications/Noise_Estimation_Single_Image.pdf <br>
[2] https://www1.cs.columbia.edu/CAVE/publications/pdfs/Grossberg_PAMI04.pdf
