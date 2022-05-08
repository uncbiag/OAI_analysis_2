import pathlib
def get_data_dir():
    return str(pathlib.Path(__file__).parents[1] / "data")


