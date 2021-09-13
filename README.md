Installation of dependencies and testing

```
git clone https://github.com/HastingsGreer/oai_analysis_2
cd oai_analysis_2
conda env create --file environment.yml
conda activate oai-analysis-2
python -m unittest -v discover
```

Currently, this should declare that the segmentation test passed, and the registration test failed.

To view the demo notebooks:
```
cd notebooks
jupyter notebook
```
