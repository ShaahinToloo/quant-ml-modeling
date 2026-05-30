# quant-ml-modeling
Python Open-Source Ai/ML modeling pipline. Implementations using PyTorch, PyTorch Lightning, Scikit-Learn.\
This project's mainly focus is on financial market's Peaks (Highs) and Troughs (Lows) detection. Hence it is revisable for other purposes.

## Requirements
If you don't have a venv. Make sure you create one using
**And make sure you are in the project's folder**
```bash
python -m venv .venv
```


Install the required libraries for working with scikit-learn via
```bash
pip install -r requirements_sklearn.txt
```

Install the required libraries for working with torch via
```bash
pip install -r requirements_torch.txt
```


## Run
To run the project go to run folder.

For scikit-learn implementations and codeflow go to:\
`run/sklearn/`

For PyTorch implementations and codeflow go to:\
`run/torch/`

There you can see the flow for peak models and trough models.

Converting to ONNX is available only for torch models, not sklearn.\
if onnx doesn't work change the opset_version.

> [!CAUTION]
> When you need to convert a model to onnx, MAKE SURE your DO NOT add conditions in 'def forward' of the model.
> Why? Because ONNX runs the model and traces it's paths of calculation.
> So if you add conditions like '''If X[0] > 16: ...''' in there, then 💀.