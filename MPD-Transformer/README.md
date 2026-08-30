## Table of contents

```text
Cross-models comparision/
  cross_model_inference_example.py
MPD-Transformer application/
  oracle_inference_example.py
  oracle_downsample_upsample_example.py
datasets/
  cross_models_test/    
  oracle_examples/      
  testset/              
modules/
  cross_models/         
  mpd_transformer_oracle/
  visualization/        
requirements.txt
install.py
```

## Install Dependencies

64-bit Python 3.10 is recommended. Navigate to this directory, then run the following command in an activated Python virtual environment or Conda environment:

```powershell
python install.py
```

install.py reads requirements.txt from the same directory and installs the PyTorch, NumPy, and Matplotlib dependencies required to run the examples into the current Python environment. Alternatively, run:

```powershell
python -m pip install -r requirements.txt
```

## demo

```powershell
python "MPD-transformer_inference_example.py"
```

Cross-model comparison example:

```powershell
python "Cross-models comparision\cross_model_inference_example.py"
```
Due to the large file size, the datasets are available at: https://cloud.tsinghua.edu.cn/d/cca73bc3fd8c433cb1fc/

In Jupyter, Spyder, or VS Code, you can run the scripts section by section using the # %% markers, or copy each section directly into a Notebook. Both Oracle scripts use canonical index 22010 by default. The available demonstration samples are 3500, 8000, and 22010




