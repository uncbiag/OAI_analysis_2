[<img src="https://github.com/uncbiag/OAI_analysis_2/actions/workflows/selfhosted-action.yml/badge.svg">](https://github.com/uncbiag/OAI_analysis_2/actions)
[<img src="https://github.com/uncbiag/OAI_analysis_2/actions/workflows/github-hosted-action.yml/badge.svg">](https://github.com/uncbiag/OAI_analysis_2/actions)

Installation of dependencies and testing

```
git clone https://github.com/HastingsGreer/oai_analysis_2
cd oai_analysis_2
pip install -r requirements.txt
python -m unittest -v discover
```

Currently, this should declare that the segmentation test passed, and the registration test failed.

To view the demo notebooks:
```
cd notebooks
jupyter notebook
```


upload test data to https://data.kitware.com/#collection/6205586c4acac99f42957ac3/folder/620559344acac99f42957d63
