from pathlib import Path

import pooch

data_dir = Path(__file__).resolve().parent / "data"
data_dir.mkdir(exist_ok=True)

github_release_tag = 'v2.0.0'
test_data_sha256 = "bfb5d5f17ff0886f5815c79fba119fc4294c35b8f9fcd586a171d9310cd90cdf"
atlases_sha256 = "9332f6756efcc6f525a8cf1807cfe793405181047f850d68fd962ad8e5ac1d7a"
models_sha256 = "ffd081ba26f9908f17790ecdb8b1d025bff32ccbdced9bf55158166561064458"

data_fetcher = pooch.create(
    path=data_dir,
    base_url=f"https://github.com/uncbiag/OAI_analysis_2/releases/download/{github_release_tag}/",
    registry={
        "oai-analysis-test-data.tar.gz": f"sha256:{test_data_sha256}",
        "oai-analysis-atlases.tar.gz": f"sha256:{atlases_sha256}",
        "oai-analysis-models.tar.gz": f"sha256:{models_sha256}",
    },
    retry_if_failed=5,
)

def test_data_dir():
    extract_dir = "test_data"
    result = data_dir / extract_dir

    untar = pooch.Untar(extract_dir=extract_dir)
    data_fetcher.fetch("oai-analysis-test-data.tar.gz", processor=untar)

    return result

def atlases_dir():
    extract_dir = "atlases"
    result = data_dir / extract_dir

    untar = pooch.Untar(extract_dir=extract_dir)
    data_fetcher.fetch("oai-analysis-atlases.tar.gz", processor=untar)

    return result

def models_dir():
    extract_dir = "models"
    result = data_dir / extract_dir

    untar = pooch.Untar(extract_dir=extract_dir)
    data_fetcher.fetch("oai-analysis-models.tar.gz", processor=untar)

    return result
