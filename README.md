# cs4375

Create or open google colab to run the code

### Installation
import os
# Set the working directory to /content
os.chdir("/content")
print("Current directory:", os.getcwd())

!git clone https://github.com/SusanLL/CS4375_HW3.git /content/CS4375_HW3

Install python packages
   ```Shell
   pip install -r requirement.txt
   ```
**Run the Code**
%run /content/CS4375_HW3/yolo/data.py
%run /content/CS4375_HW3/yolo/modelpy
%run /content/CS4375_HW3/yolo/loss.py
%run /content/CS4375_HW3/yolo/train.py
%run /content/CS4375_HW3/yolo/test.py
