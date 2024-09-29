# molmotest
Tests using molmo ai

https://molmo.allenai.org/

The IPYNB is where I wrote my tests. 

This code is useful if your model needs to use the CPU. 

The example on the allenai/molmo huggingface page is otherwise non-compatible and needed to be reworked as it uses the GPU. 

My laptop is an intel i7 with no support for using my GPU (not amd, nvdia drivers, not BnB nor Cuda compatible). 

`molmo.py` is a standalone example. 

`python molmo.py coconutladenswallow.png "point to me the coconut laden swallow"`

Saves the image with a dot on it into the molmo_images folder and then logs:


    (.venv) carlos@karen:~/Documents/GitHub/aiexp$ python molmo.py coconutladenswallow.jpg "point to me the coconut laden swallow"
    2024-09-29 13:39:42.408570: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-09-29 13:39:42.597789: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-09-29 13:39:44,276 - INFO - Loading image and models...
    2024-09-29 13:39:44,288 - INFO - Saved image: molmo_images/original_20240929_133944.jpg
    Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00,  8.22it/s]
    2024-09-29 13:39:47,237 - INFO - Processing image and question...
    2024-09-29 13:39:47,259 - INFO - Generating output...
    2024-09-29 13:42:20,892 - INFO - Processing generated output...
    2024-09-29 13:42:20,908 - INFO - Drawing points on image...
    2024-09-29 13:42:20,944 - INFO - Saved image: molmo_images/modified_20240929_134220.jpg
    Generated output:  <point x="42.0" y="27.0" alt="me the coconut laden swallow">me the coconut laden swallow</point>
