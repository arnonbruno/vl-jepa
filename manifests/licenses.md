# Annotation sources

Manifests are id lists. Image files were not downloaded.

- Karpathy split file: http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
  Local zip, `dataset_coco.json`, and `dataset_flickr30k.json` are hashed in `provenance.json`.
- COCO 2017 captions: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  Used to compare `cocoid` with train2017 and val2017.
  Terms: https://cocodataset.org/#termsofuse
- Flickr30k captions are the Karpathy split in that zip. This directory does not relicense them.
- SugarCrepe++, Winoground, GQA, Visual Genome, and Oxford-IIIT Pets were not loaded.
