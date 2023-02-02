#!/usr/bin/env python3

import coiled

# 1st Step
coiled.create_software_environment(
   name="oai-analysis-2",
   container="pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime",
   conda="environment.yml",
   post_build="post_build.sh",
)
