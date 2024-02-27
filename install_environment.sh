conda install pip=23.3.2
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.1_cu121.html
pip install diffusers==0.25.0
pip install transformers[sentencepiece]==4.36.2
pip install accelerate==0.27.0 # optional, but increases model loading speed
pip install pyrallis==0.3.1
pip install xatlas==0.0.7
pip install loguru==0.7.2
