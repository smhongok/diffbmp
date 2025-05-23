# Circle Art Test

## Overview

This project provides tools and methodologies for creating and evaluating circle-based vector art from images. It includes initialization techniques, rendering methods, and evaluation scripts aimed at comparing the effectiveness and quality of various circle-art generation algorithms.

## Project Structure

```
circle_art_test/
├── assets/
│   ├── font/
│   │   └── arial.ttf
│   └── svg/
│       ├── arial_S.svg
│       ├── bat.svg
│       └── siggraph_logo.svg
├── configs/
│   └── default.json
├── core/
│   ├── preprocessing.py
│   ├── initializer/
│   │   ├── base_initializer.py
│   │   ├── multilevel_initializer.py
│   │   ├── random_initializater.py
│   │   ├── singlelevel_initializer.py
│   │   ├── svgsplat_initializater.py
│   │   └── tm_initializer.py
│   └── renderer/
│       ├── mse_renderer.py
│       └── vector_renderer.py
├── images/
│   ├── artwork/
│   ├── BSDS500/
│   ├── CelebA/
│   ├── MoviePosters/
│   ├── MoviePosters_2/
│   ├── nature/
│   └── supp/
├── util/
│   ├── pdf_exporter.py
│   ├── run_wavelet.py
│   ├── svg_converter.py
│   ├── svg_loader.py
│   ├── svg_to_hollow.py
│   └── utils.py
├── compare_methods.py
├── main.py
├── requirements.txt
└── run_evaluation.py
```

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

* Predefined SVG templates are provided in the `assets/svg` directory.
* Fonts for rendering are located in `assets/font`.

## Examples and Datasets

The `images` directory contains various datasets and sample images categorized for quick testing:

* Artwork
* Nature
* Movie Posters
* Benchmark images (BSDS500, CelebA)

## Contributing

Feel free to submit pull requests or report issues to enhance the functionality or resolve problems.

## License

Specify the appropriate license for the use or distribution of this project.

---

Enjoy creating beautiful circle-based artwork!

