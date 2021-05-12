# Semantic Segmentation on RGB images with Unet architecture
## Execution
*Note: Since I run this model inside Anaconda environment, run.sh is made for usage inside Anaconda, using Linux built-in python would require modification. Also, I haven't test this build on Tensorflow < 2 so the code might have some error.

Step 1: Activate Anaconda.

Step 2: Check requirements in text file.

Step 3: Run bash file inside "/project" directory.

## Comments
Personally for my build, the model performance is quite poor on RGB images, cost alot of time to train due to high memory consumption. I used Sparse Catagorical Cross Entropy and Oxford IIIT Pet dataset with image scale 256x256x3, if there are any improvements on model with different tweeks in code, loss or change in architecture please let me know. Thank you.
