# TEXT-TO-HANDWRITING-GENERATOR
# Handwriting Image Generator

This project provides a tool for generating images with custom text in a specified font. By specifying a font file, text, and font size, users can generate a personalized handwriting-style image.

## Features

- **Text-to-Image Rendering**: Converts custom text into an image using a specified font.
- **Font Customization**: Supports different font styles by loading .ttf font files.
- **Scalable Output**: Adjust font size and text to meet various output needs.

## Requirements

- Python 3.x
- [Pillow](https://python-pillow.org/) - for image processing
- [TensorFlow](https://www.tensorflow.org/) and [NumPy](https://numpy.org/) (though optional unless used specifically in the project)

Install dependencies via pip:

```bash
pip install pillow tensorflow numpy
