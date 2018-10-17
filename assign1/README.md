## Practice Problems                                                                                     
All these problems should be completed using python, numpy, opencv, and matplotlib.                     
                                                                                                        
### Digital Image Structure                                                                              
    1. Read an image as RGB and display it on the screen (use opencv, matplotlib)                       
    
    2. Read an image as grayscale and display it                                                        
    
    3. Read an image as RGB and convert it to grayscale, display it                                     
    
    4. Read an image as RGB and separate the image’s color channels (use split() or numpy indexing - 
       Which method is computationally faster?).                                                    
        a. After separating the channels, make 3 new RGB images and display them                        
        b. The first new RGB image should have only the red channel (no blue/green)                     
        c. The second new RGB image should have only the green channel (no red/blue)                    
        d. The third new RGB image should have only the blue channel (no red/green)                     
    
    5. Cut out a rectangular section of the image and display it on screen (use numpy indexing)         
                                                                                                        
### Digital Image Characteristics                                                                        
    1. Read an image as grayscale and plot the histogram of its intensities (use matplotlib)            
    
    2. Read two images as grayscale and plot both of their histograms on the same plot with different 
       colors (use matplotlib)                                                                       
                                                                                                        
### Morphological Operations                                                                             
These operations are used to modify an image’s shape. Often, grayscale images are converted             
to binary images using thresholding. Thresholded images are often used for contouring and               
extracting shape from a complex image. They’re also used for finding objects in the image.              
    
    1. Save the above image and perform binary thresholding on the image to separate the                
       particles from the background. Play around with a binary threshold to find the best                 
       value or use a histogram to determine the best threshold value.                                     
    
    2. Read about morphological operations on OpenCV. On the binary image, apply erosion,               
      dilation, opening, and closing to the image to create 4 new images.                                 
      a. Play with different windows/structuring elements                                                 
      b. Display the original image and the 4 new images after applying these operations.                 
      	 What did each of the operations do to the image?                                                    
      c. After applying the operations to the image, determine which approach is best                     
         used to separate each particle. Hint: dilation is not useful for this goal.                         
    
    3. After determining the best operation to separate the particles in the binary image, apply        
       connected component labeling (CCL) to count the number of particles. Unfortunately,                 
       CCL isn’t documented for python but it is implemented. Your estimate does not need to               
       be exact.       
