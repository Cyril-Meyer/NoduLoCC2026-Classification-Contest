# NoduLoCC2026-Classification-Contest
My participation to [NoduLoCC2026 : International Lung Nodule Localization and Classification Contest from Chest X-Ray Images](https://gt-i2mdp.github.io/website/nodule_challenge.html).

We only worked on the classification of image challenge, not the localization.

Weights used for the evaluation are available on Zenodo : https://zenodo.org/records/19004996


## What is the method idea ?

We use existing backbones from `tf.keras.applications`, add them a classification head a train them using cross entropy.
The best and last models are both saved.
We select best model using a part of the training dataset reserved for this usage (validation set) which correspond to random 5% of the train set and keep best model on validation loss.

We then selected the best resulting models and use them as an ensemble to classify inputs.


## How to reproduce ?

All the code has been put in jupyter notebooks.

* Use `TF2-PrepData.ipynb` to convert the original folder of images into two files (`X.npy` and `Y.npy`).
  * The image are preprocessed and store in single file for easier usage after this
* Use `TF2-Train.ipynb` to train your model.
  * A lot of different configuration are possible using hard coding (commented section and others).
  * The common parameters used to train our models
    * `batch_generator_balanced` oversample positive samples
    * `data_augmentation` is not used
    * `backbone` is not freezed
    * We train for 100 epoch, 1 epoch = `len(train)//BATCH_SIZE//2` (so 50 real epochs)
  * Changing parameters
    * Batch size = 4, 8 or 16 depending on model parameters number
* Use `TF2-Pred.ipynb` to select models to keep
* `CM-Eval.ipynb` is provided for final prediction with example on the training dataset for the organizers.
  * 4 models are used as an ensemble
  * The ensemble strategy is made to maximize detection of nodule which seems more important in a medical assisting idea.
    * Decision = average of 4 model > 25% ?


## How to use it for the evaluation part ? A.K.A. the section for the contest organizers.

As no evaluation script where provided, I wrote a single function `predict` in the notebook `CM-Eval.ipynb`.
This function take an original image as provided in the contest, preprocess it, predict the class and return a boolean with the prediction and a confidence score (with also raw predictions).


## Is it good code ?

Not at all !
The code may have errors or not be optimal.
This is the internet : do not trust without checking.


## Requirements to run the code

* TensorFlow **tensorflow==2.16.2**
* NumPy **numpy==1.26.4**

Probably some other little things.
A copy of my current `pip freeze` is available at the end of this readme, but as a lot of things are just installed on the jupyter lab instance, they are probably useless.

## Acknowledgement

Thanks to the NoduLoCC2026 challenge organizer, especially **Adnan Mustafic** which was the member we interact with to get access to the challenge dataset.

## Misc

<details>
  <summary>requirements.txt</summary>

````
absl-py==2.2.2
aeon==1.3.0
anyio==4.9.0
argon2-cffi==23.1.0
argon2-cffi-bindings==21.2.0
arrow==1.3.0
asttokens==3.0.0
astunparse==1.6.3
async-lru==2.0.5
attrs==25.3.0
babel==2.17.0
baycomp==1.0
beautifulsoup4==4.13.4
bleach==6.2.0
certifi==2025.4.26
cffi==1.17.1
charset-normalizer==3.4.1
comm==0.2.2
contourpy==1.3.2
cycler==0.12.1
debugpy==1.8.14
decorator==5.2.1
defusedxml==0.7.1
Deprecated==1.2.18
executing==2.2.0
fastjsonschema==2.21.1
flatbuffers==25.2.10
fonttools==4.58.0
fqdn==1.5.1
gast==0.6.0
google-pasta==0.2.0
grpcio==1.71.0
h11==0.16.0
h5py==3.13.0
httpcore==1.0.9
httpx==0.28.1
idna==3.10
imageio==2.37.0
ipykernel==6.29.5
ipython==9.2.0
ipython_pygments_lexers==1.1.1
isoduration==20.11.0
jedi==0.19.2
Jinja2==3.1.6
joblib==1.4.2
json5==0.12.0
jsonpointer==3.0.0
jsonschema==4.23.0
jsonschema-specifications==2025.4.1
jupyter-events==0.12.0
jupyter-lsp==2.2.5
jupyter_client==8.6.3
jupyter_core==5.7.2
jupyter_server==2.15.0
jupyter_server_terminals==0.5.3
jupyterlab==4.4.1
jupyterlab_pygments==0.3.0
jupyterlab_server==2.27.3
keras==3.9.2
keras-tcn==3.5.6
kiwisolver==1.4.8
lazy_loader==0.4
libclang==18.1.1
llvmlite==0.44.0
Markdown==3.8
markdown-it-py==3.0.0
MarkupSafe==3.0.2
matplotlib==3.10.3
matplotlib-inline==0.1.7
mdurl==0.1.2
mistune==3.1.3
ml-dtypes==0.3.2
multi_comp_matrix==0.0.3
namex==0.0.9
nbclient==0.10.2
nbconvert==7.16.6
nbformat==5.10.4
nest-asyncio==1.6.0
networkx==3.5
notebook_shim==0.2.4
numba==0.61.2
numpy==1.26.4
nvidia-cublas-cu12==12.3.4.1
nvidia-cuda-cupti-cu12==12.3.101
nvidia-cuda-nvcc-cu12==12.3.107
nvidia-cuda-nvrtc-cu12==12.3.107
nvidia-cuda-runtime-cu12==12.3.101
nvidia-cudnn-cu12==8.9.7.29
nvidia-cufft-cu12==11.0.12.1
nvidia-curand-cu12==10.3.4.107
nvidia-cusolver-cu12==11.5.4.101
nvidia-cusparse-cu12==12.2.0.103
nvidia-nccl-cu12==2.19.3
nvidia-nvjitlink-cu12==12.3.101
opencv-python==4.13.0.92
opt_einsum==3.4.0
optree==0.15.0
overrides==7.7.0
packaging==25.0
pandas==2.2.3
pandocfilters==1.5.1
parso==0.8.4
pexpect==4.9.0
pillow==11.2.1
platformdirs==4.3.7
prometheus_client==0.21.1
prompt_toolkit==3.0.51
protobuf==4.25.7
psutil==7.0.0
ptyprocess==0.7.0
pure_eval==0.2.3
pycparser==2.22
Pygments==2.19.1
pyparsing==3.2.3
python-dateutil==2.9.0.post0
python-json-logger==3.3.0
pytz==2025.2
PyYAML==6.0.2
pyzmq==26.4.0
referencing==0.36.2
requests==2.32.3
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rich==14.0.0
rpds-py==0.24.0
scikit-image==0.25.2
scikit-learn==1.6.1
scipy==1.15.2
seaborn==0.13.2
Send2Trash==1.8.3
setuptools==80.0.1
simpleitk==2.5.2
six==1.17.0
sniffio==1.3.1
soupsieve==2.7
stack-data==0.6.3
tensorboard==2.16.2
tensorboard-data-server==0.7.2
tensorflow==2.16.2
termcolor==3.1.0
terminado==0.18.1
threadpoolctl==3.6.0
tifffile==2025.6.11
tinycss2==1.4.0
tornado==6.4.2
tqdm==4.67.1
traitlets==5.14.3
types-python-dateutil==2.9.0.20241206
typing_extensions==4.13.2
tzdata==2025.2
uri-template==1.3.0
urllib3==2.4.0
wcwidth==0.2.13
webcolors==24.11.1
webencodings==0.5.1
websocket-client==1.8.0
Werkzeug==3.1.3
wheel==0.45.1
wrapt==1.17.2
```

</details>
