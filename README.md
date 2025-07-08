# Ensemble of Different models to get a Robust Audio Source Separation

First Install the requirements:

```
    pip install -r requirements.txt
```

For our implementation was necessary edit the function load_model in the installed library demucs/states.py
We added the parameter weights_only=False this to get a complete loaded model in case of Demucs.
```
    def load_model(path_or_package, strict=False):
    """Load a model from the given serialized model, either given as a dict (already loaded)
    or a path to a file on disk."""
    if isinstance(path_or_package, dict):
        package = path_or_package
    elif isinstance(path_or_package, (str, Path)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = path_or_package
            package = torch.load(path, 'cpu', weights_only=False)
```

To get a Inference

```
    python ensemble.py --input_audio ./song1/ --output_folder ./results/
```

With this command audio with name "mixture.wav" inside of song1 folder will be processed and results will be stored in `./results/` folder in WAV format.

Note: You can use mp3 and m4a files also but yo need to modify ensemble.py.

### All available keys
* `--input_audio` - input audio folder location. It's necessary to specify a path with a mixture.wav file. **Required**
* `--output_folder` - output audio folder. **Required**

