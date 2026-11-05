# Sources and licences

## Matching benchmark
https://github.com/BstnGHub/Multimodal-Product-Matching
Pinned commit: 48a8e9959c4edae0a1dd321fe195e5280422b9a9
Author repository for “Prompt or Train? A Comparative Benchmark of VLM Prompting and Intermediate Fusion for Product Matching.”
WDC Shoes files contain text/image pairs and labels. Downloads verify the Git LFS pointer SHA256 and byte size at the pinned commit. Original split listing overlap is recorded; our results use custom grouping.
The repository has an MIT software licence; that is not independent evidence of permission to relicense every underlying retailer image. This package redistributes no dataset/images. Use download scripts, credit the source, and review underlying image terms before public redistribution/commercial use.

## Amazon Berkeley Objects (ABO)
https://registry.opendata.aws/amazon-berkeley-objects/
https://amazon-berkeley-objects.s3.amazonaws.com/index.html
Metadata archive: https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar
Current official licence: CC BY 4.0 (updated from older noncommercial terms).
Credit data/images to Amazon.com and dataset construction to Matthieu Guillaumin, Thomas Dideriksen, Kenan Deng, Himanshu Arora, Jasmine Collins and Jitendra Malik.
Reference: ABO: Dataset and Benchmarks for Real-World 3D Object Understanding (CVPR 2022).
Only metadata is used by our scale command. The composite domain/item key preserves separate market listings. It is not labelled duplicate ground truth.

## Frozen encoders
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Revision e8f8c211226b894fcb81acc59f3b34ba3efd5f42
openai/clip-vit-base-patch32
Revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
Downloaded from Hugging Face; observe model-card terms. The CLIP pinned revision uses a PyTorch checkpoint loaded with weights_only=True. No trust_remote_code is enabled. Model weights are not redistributed in this ZIP.
