This project implements a simplified YOLO pipeline to detect a single “cracker box” object from 200 RGB images.

Create or open google colab to run the code
```Shell
### Installation
import os
# Set the working directory to /content
os.chdir("/content")
print("Current directory:", os.getcwd())

!git clone https://github.com/SusanLL/cracker-box-yolo.git /content/cracker-box-yolo
```
Install python packages
   ```Shell
   pip install -r requirement.txt
   ```
**Run the Code**
```Shell
%run /content/CS4375_HW3/yolo/data.py

%run /content/CS4375_HW3/yolo/modelpy

%run /content/CS4375_HW3/yolo/loss.py

%run /content/CS4375_HW3/yolo/train.py

%run /content/CS4375_HW3/yolo/test.py
```


