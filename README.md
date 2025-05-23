# Circle Art

## Overview

This project provides tools and methodologies for creating and evaluating svg-based vector art from images. It includes initialization techniques, rendering methods, and evaluation scripts aimed at comparing the effectiveness and quality of various svgsplat art generation algorithms.


## Requirements

Install dependencies using pip:

```
pip install -r requirements.txt
```

## Usage

### Running the Main Script

Execute the main script with a configuration file:

```
python main.py --config configs/default.json
```

### Evaluating Methods

To compare different circle-art methods:

```
python compare_methods.py
```

### Running Evaluations

To execute specific evaluations on generated results:

```
python run_evaluation.py
```

## Assets

* Put any predefined SVG templates based on 'path' tag in the `assets/svg` directory.
* Put fonts for rendering in `assets/font`.

## Examples and Datasets

The `images` directory contains various datasets and sample images categorized for quick testing:

* Artwork
* Nature
* Movie Posters
* Benchmark images (BSDS500, CelebA)

## Contributing

Feel free to submit pull requests or report issues to enhance the functionality or resolve problems.

## License

Please do not distribute. This is for the purpose of anonymous review.

---

Enjoy creating beautiful SVG drawing artwork!

