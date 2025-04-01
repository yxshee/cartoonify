# Cartoonify 🎨


**Cartoonify** is a Python-based application designed to transform your everyday photos into stunning cartoon-like images. By leveraging computer vision and image processing techniques such as edge detection, color quantization, and smoothing filters, Cartoonify provides artistic and visually appealing results.

---

## 🚀 Features

- **Simple Cartoonification:** Convert regular images into cartoon-like representations effortlessly.
- **Adjustable Settings:** Customize the level of cartoon effect, including edge thickness, color depth, and smoothness.
- **Batch Processing:** Handle multiple images simultaneously, improving efficiency and productivity.

<img width="1279" alt="image" src="https://github.com/user-attachments/assets/f06655a9-94f9-429c-b141-51694a33cf77" />


---

## 🖼️ Sample Results

Take a look at the transformation:

| Original Image | Cartoonified Image |
|----------------|--------------------|
| <img width="238" alt="image" src="https://github.com/user-attachments/assets/476bb0d6-5cb8-4fb1-833e-61ecae7cf409" /> | <img width="238" alt="image" src="https://github.com/user-attachments/assets/7c5b2d74-c602-41f4-a599-cac42ad29621" />|




---

## 💻 Installation

Follow these steps to set up Cartoonify on your machine:

### Step 1: Clone the Repository

```bash
git clone https://github.com/yxshee/cartoonify.git
```

### Step 2: Navigate to the Project Directory

```bash
cd cartoonify
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📌 Usage

### Prepare Your Images

Place your images in the `data/input` folder.

### Run the Cartoonify Script

```bash
python src/cartoonify.py --input_dir data/input --output_dir data/output
```

- `--input_dir`: Path to your original images.
- `--output_dir`: Path to save cartoonified images.

### View Your Results

Cartoonified images will be saved in the `data/output` directory.

---

## 📂 Project Structure

```
cartoonify/
├── data/
│   ├── input/
│   └── output/
├── docs/
├── src/
│   ├── cartoonify.py
│   └── utils.py
├── tests/
├── assets/
├── requirements.txt
└── README.md
```

- `data/input/`: Input images.
- `data/output/`: Cartoonified output images.
- `src/`: Source code.
- `tests/`: Unit tests.
- `docs/`: Documentation.
- `assets/`: Images and media files for documentation.

---

## 📦 Dependencies

Cartoonify requires the following libraries:

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

You can install these using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions and improvements are welcome!

- Fork the repository.
- Create your feature branch (`git checkout -b feature/AmazingFeature`).
- Commit your changes (`git commit -m 'Add some AmazingFeature'`).
- Push to the branch (`git push origin feature/AmazingFeature`).
- Open a Pull Request.

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](https://github.com/yxshee/cartoonify/blob/main/LICENSE) for details.

---

## 🙏 Acknowledgments

- Thanks to the OpenCV community and open-source contributors.



